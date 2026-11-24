"""Model calls: read the note into formal terms, then choose codes from the search candidates."""
import os
import re
import sys
import threading
import time
import types

# Each step has a model preference list. A model the key
# cannot use is dropped on its first 404, and a provider with no models left hands over to the next one.
ANTHROPIC_READ = ["claude-haiku-5-5", "claude-haiku-4-5", "claude-sonnet-5-5", "claude-sonnet-4-5"]
ANTHROPIC_RANK = ["claude-sonnet-5-5", "claude-sonnet-4-5", "claude-haiku-5-5", "claude-haiku-4-5"]
OPENAI_READ = ["gpt-6-luna", "gpt-5-mini", "gpt-6.1-terra", "gpt-5"]
OPENAI_RANK = ["gpt-6.1-terra", "gpt-5", "gpt-6-luna", "gpt-5-mini"]
RANK_EFFORT = "low"
# US$ per token, used by the spend guard. Anthropic 5.5 list prices cached 2026-10-06, GPT-6 read 2026-11-07,
# older fallbacks from their published list prices. A model not listed here is priced high on purpose.
PRICES = {"claude-haiku-5-5": (0.10e-6, 0.50e-6), "claude-sonnet-5-5": (2e-6, 10e-6),
          "claude-haiku-4-5": (1e-6, 5e-6), "claude-sonnet-4-5": (3e-6, 15e-6),
          "gpt-6-luna": (0.10e-6, 0.50e-6), "gpt-6.1-terra": (1e-6, 5e-6),
          "gpt-5-mini": (0.25e-6, 2e-6), "gpt-5": (1.25e-6, 10e-6)}
UNKNOWN_PRICE = (5e-6, 25e-6)
# A model that rejects this many requests before ever answering is treated as unusable with our requests.
MAX_REJECTS = 3
MAX_USD = 4.50
CHEAP_RANK_FROM_USD = 3.75
MAX_MINUTES = 50
MAX_NOTE_CHARS = 16000
# An outage pauses the calls rather than ending them: the run has far more time than it needs.
MAX_FAILURES_IN_A_ROW = 25
PAUSE_SECONDS = 60
MAX_PAUSES = 6
# Timeout for a single SDK request.
CALL_TIMEOUT = 60

READ_PROMPT = """You read outpatient clinical notes for a medical coding team. The note may be in any language, and may be a typed note, shorthand, or a transcript of a consultation.

<note>
{note}
</note>

Work out what this visit should be coded as. Keep only conditions the clinician confirms or treats as the working diagnosis. Leave out anything ruled out, only queried, or mentioned in passing and not addressed. If no diagnosis is stated, give the main symptom, finding or reason for the visit.

Reply in English using formal medical terminology, as it would appear in a disease classification. Spell out abbreviations. Include every detail that would change the code: type, cause, site, severity, stage, acute or chronic, complications, pregnancy. Do not add detail the note does not give.

Format, with no other text:
MAIN: <the main condition with its details>
ALT: <another formal name for the main condition, if there is one>
OTHER: <one other addressed condition per OTHER line, if any>"""

RANK_PROMPT = """You assign ICD-11 codes to outpatient visits so that insurance claims are accepted.

<note>
{note}
</note>

<summary>
{read}
</summary>

Candidate codes:
{options}

Choose the three candidates most likely to be the correct code for this visit, best first. The note is the evidence; the summary is a helper and may be wrong.

- Code the main reason for the visit, not a side issue or a background condition.
- Choose the most specific code the note supports. Use a named subtype when the note gives that detail.
- Use an unspecified code (usually ending in Z) when the note does not give the detail a subtype needs. Do not assume detail.
- Use an "other specified" code (usually ending in Y) only when the note names a specific type that has no code of its own.
- A symptom code is right only when the visit ends with no diagnosis.

Reply with one final line and no other text:
CODES: <code>, <code>, <code>"""

CODE = re.compile(r"\b[0-9A-Z]{4}(?:\.[0-9A-Z]{1,2})?\b")


def clip(text):
    """Keep the start and end of an extreme note, since the diagnosis and plan sit at the end."""
    if len(text) <= MAX_NOTE_CHARS:
        return text
    half = MAX_NOTE_CHARS // 2
    return text[:half] + "\n[...]\n" + text[-half:]


def parse_codes(answer):
    line = answer.rsplit("CODES:", 1)[-1]
    return CODE.findall(line.upper())


def out_of_credit(error):
    """An empty account cannot recover during the run, so it ends that provider like a bad key."""
    return getattr(error, "code", None) == "insufficient_quota" or "credit balance" in str(error).lower()


class Provider:
    """One API account: its client, the errors that end it, and the models still worth trying."""

    def __init__(self, name, client, fatal, missing, rejected, read_models, rank_models):
        self.name = name
        self.client = client
        self.fatal = fatal          # bad key or no permission: the whole account is unusable
        self.missing = missing      # 404: this key cannot use this model
        self.rejected = rejected    # 400: this model does not accept the request as we send it
        self.read_models = list(dict.fromkeys(read_models))
        self.rank_models = list(dict.fromkeys(rank_models))
        self.dead = None
        self.worked = set()
        self.rejects = {}
        self.plain = set()          # models that refused the effort setting, called without it

    def model(self, kind, cheap=False):
        if kind == "read" or cheap:
            return (self.read_models or self.rank_models)[0]
        return (self.rank_models or self.read_models)[0]

    def drop(self, model, why):
        for models in (self.read_models, self.rank_models):
            if model in models:
                models.remove(model)
        print(f"{self.name}: not using {model} ({why})", file=sys.stderr, flush=True)
        if not self.read_models and not self.rank_models:
            self.kill("no usable model left")

    def kill(self, why):
        if not self.dead:
            self.dead = why
            print(f"{self.name}: stopped ({why})", file=sys.stderr, flush=True)

    def reject(self, model, error, effort):
        """Handle a 400. Return True when the call is worth repeating with a changed setup."""
        text = str(error).lower()
        if effort and ("effort" in text or "reasoning" in text):
            # Calls already in flight with the effort setting land here too, after the switch.
            if model not in self.plain:
                self.plain.add(model)
                print(f"{self.name}: calling {model} without the effort setting", file=sys.stderr, flush=True)
            return True
        if model in self.worked:
            return False  # the model works, so this note's request is the problem
        self.rejects[model] = self.rejects.get(model, 0) + 1
        if self.rejects[model] >= MAX_REJECTS:
            self.drop(model, f"{MAX_REJECTS} requests rejected")
            return True
        return False


class Llm:
    def __init__(self, providers, effort=None, max_usd=None, max_minutes=None):
        self.providers = providers
        self.effort = effort or os.environ.get("CODER_RANK_EFFORT", RANK_EFFORT)
        self.max_usd = max_usd if max_usd is not None else float(os.environ.get("CODER_MAX_USD", MAX_USD))
        minutes = max_minutes if max_minutes is not None else float(os.environ.get("CODER_MAX_MINUTES", MAX_MINUTES))
        self.deadline = time.time() + minutes * 60
        self.lock = threading.Lock()
        self.spent = 0.0
        self.tokens = {}
        self.calls = 0
        self.failures = 0
        self.pauses = 0
        self.paused_until = 0.0
        self.stopped = None

    def provider(self):
        return next((p for p in self.providers if not p.dead), None)

    def describe(self):
        live = [p for p in self.providers if not p.dead]
        if not live:
            return "no provider"
        first = live[0]
        text = f"{first.name}: {first.model('read')} then {first.model('rank')}"
        return text + "".join(f", then {p.name} if needed" for p in live[1:])

    def usable(self):
        if self.stopped:
            return False
        if self.spent >= self.max_usd:
            self.stop("spend guard reached")
        elif time.time() >= self.deadline:
            self.stop("time guard reached")
        return not self.stopped

    def stop(self, reason):
        with self.lock:
            if not self.stopped:
                self.stopped = reason
                print(f"model calls stopped: {reason}; remaining notes use search only", file=sys.stderr, flush=True)

    def failed(self):
        with self.lock:
            self.failures += 1
            if self.failures < MAX_FAILURES_IN_A_ROW:
                return
            self.failures = 0
            self.pauses += 1
            give_up = self.pauses > MAX_PAUSES
            if not give_up:
                self.paused_until = time.time() + PAUSE_SECONDS
                print(f"{MAX_FAILURES_IN_A_ROW} failures in a row; pausing {PAUSE_SECONDS}s (pause {self.pauses})",
                      file=sys.stderr, flush=True)
        if give_up:
            self.stop(f"still failing after {MAX_PAUSES} pauses")

    def ask(self, kind, prompt, effort, max_tokens, cheap=False):
        """Return the reply text, or None so that the caller can fall back to search.

        Every repeat of the loop first drops a model, ends a provider or turns off the effort
        setting for a model, so it always finishes.
        """
        while True:
            while self.usable() and time.time() < self.paused_until:
                time.sleep(0.5)
            if not self.usable():
                return None
            with self.lock:
                p = self.provider()
                model = p.model(kind, cheap) if p else None
            if not p:
                self.stop("no usable provider (" + "; ".join(f"{q.name}: {q.dead}" for q in self.providers) + ")")
                return None
            use_effort = effort if model not in p.plain else None
            request = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
            if use_effort:
                request["output_config"] = {"effort": use_effort}
            try:
                r = p.client.messages.create(**request)
            except p.fatal as e:
                with self.lock:
                    p.kill(f"{type(e).__name__} from the API")
                continue
            except p.missing:
                with self.lock:
                    if model in p.read_models or model in p.rank_models:
                        p.drop(model, "not available to this key")
                continue
            except Exception as e:
                if out_of_credit(e):
                    with self.lock:
                        p.kill("account out of credit")
                    continue
                if isinstance(e, p.rejected):
                    with self.lock:
                        again = p.reject(model, e, use_effort)
                    if again:
                        continue
                    print(f"model call rejected: {type(e).__name__}", file=sys.stderr, flush=True)
                    return None
                print(f"model call failed: {type(e).__name__}", file=sys.stderr, flush=True)
                self.failed()
                return None
            price_in, price_out = PRICES.get(model, UNKNOWN_PRICE)
            with self.lock:
                p.worked.add(model)
                self.failures = 0
                self.pauses = 0
                self.calls += 1
                self.spent += r.usage.input_tokens * price_in + r.usage.output_tokens * price_out
                used = self.tokens.setdefault(model, [0, 0])
                used[0] += r.usage.input_tokens
                used[1] += r.usage.output_tokens
            if r.stop_reason in ("refusal", "max_tokens"):
                return None
            return "".join(b.text for b in r.content if b.type == "text").strip() or None

    def read(self, note):
        return self.ask("read", READ_PROMPT.format(note=clip(note)), "low", 3000)

    def rank(self, note, read, options):
        prompt = RANK_PROMPT.format(note=clip(note), read=read, options=options)
        # Near the guard, a cheaper ranking beats dropping the remaining notes to search only.
        cheap = self.spent >= self.max_usd * CHEAP_RANK_FROM_USD / MAX_USD
        answer = self.ask("rank", prompt, self.effort, 6000, cheap=cheap)
        return parse_codes(answer) if answer else []

    def report(self):
        return f"{self.calls} model calls, tokens {self.tokens}, about US${self.spent:.3f}"


class OpenAIClient:
    """Gives the OpenAI client the same call and reply shape as the Anthropic one, so the guards cover both."""
    STOPS = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}

    def __init__(self, client):
        self.client = client
        self.messages = self

    def create(self, model, max_tokens, messages, output_config=None):
        extra = {"reasoning_effort": output_config["effort"]} if output_config else {}
        r = self.client.chat.completions.create(model=model, max_completion_tokens=max_tokens, messages=messages,
                                                **extra)
        choice = r.choices[0]
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=choice.message.content or "")],
            stop_reason=self.STOPS.get(choice.finish_reason, "end_turn"),
            usage=types.SimpleNamespace(input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens))


def has_key(name):
    return bool(os.environ.get(name, "").strip())


def timeout():
    return float(os.environ.get("CODER_CALL_TIMEOUT", CALL_TIMEOUT))


def models(kind, defaults):
    """A model named in the environment goes first, ahead of the built-in list."""
    chosen = os.environ.get(f"CODER_{kind}_MODEL", "").strip()
    return [chosen, *defaults] if chosen else list(defaults)


def connect():
    """Return an Llm over every provider with a key and an SDK, Anthropic first, or None for search only.

    The US$ guard counts spend across all of them, since the cap covers the whole run.
    """
    providers = []
    if has_key("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            providers.append(Provider(
                "anthropic", anthropic.Anthropic(max_retries=3, timeout=timeout()),
                (anthropic.AuthenticationError, anthropic.PermissionDeniedError), anthropic.NotFoundError,
                anthropic.BadRequestError, models("READ", ANTHROPIC_READ), models("RANK", ANTHROPIC_RANK)))
        except ImportError:
            print("anthropic package missing", file=sys.stderr)
    if has_key("OPENAI_API_KEY"):
        try:
            import openai
            providers.append(Provider(
                "openai", OpenAIClient(openai.OpenAI(max_retries=3, timeout=timeout())),
                (openai.AuthenticationError, openai.PermissionDeniedError), openai.NotFoundError,
                openai.BadRequestError, models("READ", OPENAI_READ), models("RANK", OPENAI_RANK)))
        except ImportError:
            print("openai package missing", file=sys.stderr)
    return Llm(providers) if providers else None
