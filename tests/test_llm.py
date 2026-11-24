import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from coder.__main__ import code_note  # noqa: E402
from coder import llm as llm_module  # noqa: E402
from coder.llm import MAX_NOTE_CHARS, MAX_REJECTS, Llm, OpenAIClient, Provider, clip, connect, parse_codes  # noqa: E402
from coder.search import Index, fuse, read_lines  # noqa: E402

INDEX = Index(os.path.join(ROOT, "data", "who", "codes.jsonl.gz"), os.path.join(ROOT, "data", "shorthand.json"))
NOTE = "Fever and sore throat. Dx acute tonsillitis."


class BadKey(Exception):
    pass


class Missing(Exception):
    pass


class Rejected(Exception):
    pass


class NoCredit(Exception):
    code = "insufficient_quota"


class Client:
    """Stands in for the SDK client. Each reply is a string, or an exception to raise."""

    def __init__(self, *replies, tokens=(100, 10)):
        self.replies = list(replies)
        self.tokens = tokens
        self.sent = []
        self.messages = self

    def create(self, **kwargs):
        self.sent.append(kwargs["model"])
        self.kwargs = kwargs
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=reply)], stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=self.tokens[0], output_tokens=self.tokens[1]))


def provider(client, name="test", read=("claude-haiku-5-5",), rank=("claude-sonnet-5-5",)):
    return Provider(name, client, (BadKey,), Missing, Rejected, list(read), list(rank))


def llm(client, read_model="claude-haiku-5-5", rank_model="claude-sonnet-5-5", providers=None, **kwargs):
    kwargs.setdefault("effort", "low")
    kwargs.setdefault("max_usd", 1.0)
    kwargs.setdefault("max_minutes", 5)
    return Llm(providers or [provider(client, read=[read_model], rank=[rank_model])], **kwargs)


class FakeOpenAI:
    def __init__(self, finish="stop", text="MAIN: Gout"):
        self.reply = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason=finish, message=types.SimpleNamespace(content=text))],
            usage=types.SimpleNamespace(prompt_tokens=1000, completion_tokens=100))
        self.sent = None
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.sent = kwargs
        return self.reply


class OpenAI(unittest.TestCase):
    def test_call_and_reply_are_translated(self):
        fake = FakeOpenAI()
        model = llm(OpenAIClient(fake), read_model="gpt-6-luna", rank_model="gpt-6.1-terra")
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual((fake.sent["model"], fake.sent["reasoning_effort"]), ("gpt-6-luna", "low"))
        self.assertEqual(fake.sent["max_completion_tokens"], 3000)
        self.assertNotIn("max_tokens", fake.sent)
        self.assertAlmostEqual(model.spent, 1000 * 0.10e-6 + 100 * 0.50e-6)

    def test_cut_off_reply_counts_as_no_answer(self):
        self.assertIsNone(llm(OpenAIClient(FakeOpenAI(finish="length"))).read(NOTE))

    def test_empty_reply_counts_as_no_answer(self):
        self.assertIsNone(llm(OpenAIClient(FakeOpenAI(text=None))).read(NOTE))

    def test_no_keys_means_search_only(self):
        saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        try:
            self.assertIsNone(connect())
        finally:
            os.environ.update({k: v for k, v in saved.items() if v})


class Fallback(unittest.TestCase):
    """No provider or model is guaranteed at grading."""

    def test_missing_model_moves_to_the_next_one(self):
        client = Client(Missing(), "MAIN: Gout", "MAIN: Gout")
        p = provider(client, read=["haiku-a", "haiku-b"], rank=["sonnet", "haiku-a", "haiku-b"])
        model = llm(None, providers=[p])
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(client.sent, ["haiku-a", "haiku-b", "haiku-b"])
        self.assertEqual(p.rank_models, ["sonnet", "haiku-b"])
        self.assertEqual(model.failures, 0)

    def test_no_models_left_hands_over_to_the_next_provider(self):
        first, second = Client(Missing()), Client("MAIN: Gout")
        model = llm(None, providers=[provider(first, "a", ["m1"], ["m2"]), provider(second, "b", ["x"], ["y"])])
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(first.sent, ["m1", "m2"])
        self.assertEqual(second.sent, ["x"])
        self.assertIsNone(model.stopped)

    def test_bad_key_hands_over_to_the_next_provider(self):
        first, second = Client(BadKey()), Client("MAIN: Gout")
        model = llm(None, providers=[provider(first, "a"), provider(second, "b")])
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(len(first.sent), 1)

    def test_empty_account_hands_over_to_the_next_provider(self):
        first, second = Client(NoCredit()), Client("MAIN: Gout")
        model = llm(None, providers=[provider(first, "a"), provider(second, "b")])
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertIn("credit", model.providers[0].dead)

    def test_every_provider_failing_means_search_only(self):
        model = llm(None, providers=[provider(Client(BadKey()), "a"), provider(Client(Missing()), "b")])
        self.assertEqual(code_note(NOTE, INDEX, model), INDEX.codes(NOTE)[:3])
        self.assertIn("no usable provider", model.stopped)

    def test_spend_is_counted_across_providers(self):
        first, second = Client("MAIN: x", tokens=(1_000_000, 0)), Client("MAIN: y", tokens=(1_000_000, 0))
        model = llm(None, providers=[provider(first, "a", ["gpt-6-luna"], ["gpt-6-luna"]),
                                     provider(second, "b", ["claude-haiku-5-5"], ["claude-haiku-5-5"])])
        model.read(NOTE)
        model.providers[0].kill("test")
        model.read(NOTE)
        self.assertAlmostEqual(model.spent, 0.2)

    def test_unknown_model_is_priced_high(self):
        model = llm(Client("MAIN: x", tokens=(1_000_000, 0)), read_model="some-new-model")
        model.read(NOTE)
        self.assertAlmostEqual(model.spent, 5.0)

    def test_effort_refused_retries_without_it(self):
        client = Client(lambda kw: Rejected("output_config.effort: not supported") if "output_config" in kw
                        else "MAIN: Gout")
        model = llm(client)
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertNotIn("output_config", client.kwargs)
        self.assertEqual(len(client.sent), 3)

    def test_effort_refusals_already_in_flight_do_not_drop_the_model(self):
        client = Client("MAIN: Gout")
        p = provider(client, read=["a", "b"])
        model = llm(None, providers=[p])
        p.worked.add("a")
        for _ in range(MAX_REJECTS + 1):
            self.assertTrue(p.reject("a", Rejected("reasoning_effort not supported"), "low"))
        self.assertEqual(p.read_models, ["a", "b"])
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertNotIn("output_config", client.kwargs)

    def test_model_rejecting_every_request_is_dropped(self):
        client = Client(lambda kw: Rejected("bad") if kw["model"] == "a" else "MAIN: Gout")
        p = provider(client, read=["a", "b"], rank=["a", "b"])
        model = llm(None, providers=[p])
        self.assertIsNone(model.read(NOTE))
        self.assertIsNone(model.read(NOTE))
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(p.read_models, ["b"])

    def test_one_rejected_note_on_a_working_model_keeps_the_model(self):
        client = Client("MAIN: Gout", Rejected("bad"), Rejected("bad"), Rejected("bad"), "MAIN: Gout")
        p = provider(client, read=["a", "b"])
        model = llm(None, providers=[p])
        model.read(NOTE)
        for _ in range(3):
            self.assertIsNone(model.read(NOTE))
        self.assertEqual(model.read(NOTE), "MAIN: Gout")
        self.assertEqual(p.read_models, ["a", "b"])

    def test_openai_adapter_drops_effort_when_asked(self):
        fake = FakeOpenAI()
        OpenAIClient(fake).create(model="m", max_tokens=10, messages=[])
        self.assertNotIn("reasoning_effort", fake.sent)

    def test_named_model_goes_first(self):
        os.environ["CODER_RANK_MODEL"] = "my-model"
        try:
            self.assertEqual(llm_module.models("RANK", ["a"]), ["my-model", "a"])
        finally:
            del os.environ["CODER_RANK_MODEL"]


class Parsing(unittest.TestCase):
    def test_parse_codes_reads_the_final_line(self):
        self.assertEqual(parse_codes("thinking about 1A00\nCODES: CA03.Z, 1F40.0, 5A11"), ["CA03.Z", "1F40.0", "5A11"])

    def test_read_lines_ignores_empty_fields(self):
        fields = read_lines("MAIN: Gout\nALT:\nOTHER: Hypertension\nOTHER: Obesity")
        self.assertEqual((fields["MAIN"], fields["ALT"], fields["OTHER"]), (["Gout"], [], ["Hypertension", "Obesity"]))

    def test_fuse_prefers_codes_in_both_lists(self):
        self.assertEqual(fuse([(1, ["A", "B"]), (1, ["C", "B"])])[0], "B")

    def test_clip_keeps_both_ends(self):
        text = "start" + "x" * 50000 + "end"
        kept = clip(text)
        self.assertLess(len(kept), MAX_NOTE_CHARS + 20)
        self.assertTrue(kept.startswith("start") and kept.endswith("end"))


class Pipeline(unittest.TestCase):
    def test_model_pick_comes_first(self):
        client = Client("MAIN: Acute tonsillitis", "CODES: CA03.Z, CA03.0, CA0Z")
        self.assertEqual(code_note(NOTE, INDEX, llm(client))[0], "CA03.Z")
        self.assertEqual(client.sent, ["claude-haiku-5-5", "claude-sonnet-5-5"])

    def test_api_error_falls_back_to_search(self):
        self.assertEqual(code_note(NOTE, INDEX, llm(Client(RuntimeError("503")))), INDEX.codes(NOTE)[:3])

    def test_rank_failure_keeps_the_merged_search(self):
        codes = code_note(NOTE, INDEX, llm(Client("MAIN: Acute tonsillitis", RuntimeError("503"))))
        self.assertEqual(codes[:3], INDEX.ranked(NOTE, "MAIN: Acute tonsillitis")[:3])

    def test_bad_key_stops_all_calls(self):
        client = Client(BadKey())
        model = llm(client)
        code_note(NOTE, INDEX, model)
        code_note(NOTE, INDEX, model)
        self.assertEqual(len(client.sent), 1)
        self.assertIn("BadKey", model.stopped)

    def test_empty_note_makes_no_calls(self):
        client = Client("MAIN: x")
        code_note("  ", INDEX, llm(client))
        self.assertEqual(client.sent, [])


class Guards(unittest.TestCase):
    def test_spend_guard_stops_calls(self):
        client = Client("MAIN: Acute tonsillitis", tokens=(10_000_000, 0))
        model = llm(client, max_usd=0.5)
        model.read(NOTE)
        self.assertIsNone(model.read(NOTE))
        self.assertEqual(len(client.sent), 1)
        self.assertIn("spend", model.stopped)

    def test_ranking_turns_cheap_near_the_guard(self):
        client = Client("CODES: CA03.Z")
        model = llm(client, max_usd=1.0)
        model.spent = 0.9
        model.rank(NOTE, "MAIN: x", "CA03.Z: y")
        self.assertEqual(client.sent, ["claude-haiku-5-5"])

    def test_outage_pauses_then_recovers(self):
        old = llm_module.MAX_FAILURES_IN_A_ROW, llm_module.PAUSE_SECONDS
        llm_module.MAX_FAILURES_IN_A_ROW, llm_module.PAUSE_SECONDS = 2, 0.01
        try:
            model = llm(Client(RuntimeError("down"), RuntimeError("down"), "MAIN: Acute tonsillitis"))
            self.assertIsNone(model.read(NOTE))
            self.assertIsNone(model.read(NOTE))
            self.assertEqual(model.read(NOTE), "MAIN: Acute tonsillitis")
            self.assertIsNone(model.stopped)
            self.assertEqual(model.pauses, 0)
        finally:
            llm_module.MAX_FAILURES_IN_A_ROW, llm_module.PAUSE_SECONDS = old

    def test_lasting_outage_gives_up(self):
        old = llm_module.MAX_FAILURES_IN_A_ROW, llm_module.PAUSE_SECONDS, llm_module.MAX_PAUSES
        llm_module.MAX_FAILURES_IN_A_ROW, llm_module.PAUSE_SECONDS, llm_module.MAX_PAUSES = 1, 0.01, 2
        try:
            model = llm(Client(RuntimeError("down")))
            for _ in range(3):
                model.read(NOTE)
            self.assertIn("pauses", model.stopped)
        finally:
            llm_module.MAX_FAILURES_IN_A_ROW, llm_module.PAUSE_SECONDS, llm_module.MAX_PAUSES = old

    def test_time_guard_stops_calls(self):
        client = Client("MAIN: x")
        self.assertIsNone(llm(client, max_minutes=0).read(NOTE))
        self.assertEqual(client.sent, [])


if __name__ == "__main__":
    unittest.main()
