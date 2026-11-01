#!/usr/bin/env python3
"""Chooses the model, and optionally the effort, for a Claude request in claude.yml.

The first line of the triggering comment, or of a new issue's body, may name a model
right after the trigger phrase: "@claude [sonnet] fix the build". The name is a short
name (haiku, sonnet, opus) or a full model ID, and must be in AF_CI_CLAUDE_MODELS, a
comma-separated list of full IDs. Without a name the model is AF_CI_CLAUDE_MODEL, or
claude-haiku-5-5. Nothing else in the text changes the model. AF_CI_CLAUDE_EFFORT, if
set, is passed on as --effort.

Writes decision (run or refuse), model, effort, number and reply to GITHUB_OUTPUT and
the choice to GITHUB_STEP_SUMMARY. A bad repository setting stops the job.
"""
import json
import os
import re
import secrets
import sys

SHORT_NAMES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MODELS = "claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# A full, pinned model ID: also safe to pass on as a single command-line argument.
MODEL_ID = re.compile(r"claude(?:-[a-z0-9]+)+")
OVERRIDE = re.compile(r"\s*@claude\s*\[([A-Za-z0-9.-]{1,64})\](?:\s|$)")


class SettingsError(Exception):
    """A repository variable is set to something this script won't use."""


def request_text(event):
    """The text that triggered the request: the comment, or a new issue's body."""
    if event.get("comment"):
        return event["comment"].get("body") or ""
    return (event.get("issue") or {}).get("body") or ""


def resolve(name):
    return SHORT_NAMES.get(name.lower(), name.lower())


def allowed_models(settings):
    models = [m.strip() for m in (settings["models"] or "").split(",") if m.strip()]
    for model in models or DEFAULT_MODELS.split(","):
        if not MODEL_ID.fullmatch(model):
            raise SettingsError(f"AF_CI_CLAUDE_MODELS must list full model IDs, not {model!r}")
    return models or DEFAULT_MODELS.split(",")


def choose(text, settings):
    """Returns the decision for one request text; raises SettingsError on bad settings."""
    allowed = allowed_models(settings)
    default = resolve(settings["model"].strip() or DEFAULT_MODEL)
    if default not in allowed:
        raise SettingsError(f"AF_CI_CLAUDE_MODEL {default!r} is not in AF_CI_CLAUDE_MODELS")
    effort = settings["effort"].strip().lower()
    if effort and effort not in EFFORTS:
        raise SettingsError(f"AF_CI_CLAUDE_EFFORT must be one of {', '.join(EFFORTS)}")
    names = ", ".join(f"`{n}`" for n in [*[s for s, m in SHORT_NAMES.items() if m in allowed],
                                            *[m for m in allowed if m not in SHORT_NAMES.values()]])

    first = (text.splitlines() or [""])[0]
    if not re.match(r"\s*@claude\s*\[", first):
        return {"decision": "run", "reason": "default", "model": default, "effort": effort, "reply": ""}
    override = OVERRIDE.match(first)
    if not override:
        return {"decision": "refuse", "reason": "malformed", "model": "", "effort": "",
                "reply": f"To choose a model, start with `@claude [name]`, where the name is one of {names}."}
    model = resolve(override.group(1))
    if model not in allowed:
        return {"decision": "refuse", "reason": "not-allowed", "model": "", "effort": "",
                "reply": f"The model `{override.group(1)}` isn't allowed here. Allowed: {names}."}
    return {"decision": "run", "reason": "requested", "model": model, "effort": effort, "reply": ""}


def settings_from_env():
    return {"model": os.environ.get("AF_CI_CLAUDE_MODEL", ""),
            "models": os.environ.get("AF_CI_CLAUDE_MODELS", ""),
            "effort": os.environ.get("AF_CI_CLAUDE_EFFORT", "")}


def main():
    with open(os.environ["GITHUB_EVENT_PATH"]) as file:
        event = json.load(file)
    try:
        result = choose(request_text(event), settings_from_env())
    except SettingsError as error:
        sys.exit(f"error: {error}")
    number = (event.get("issue") or event.get("pull_request") or {}).get("number", "")
    print(f"decision: {result['decision']} ({result['reason']}) model={result['model'] or '-'} "
          f"effort={result['effort'] or '-'}")
    delimiter = f"END_{secrets.token_hex(16)}"
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key in ("decision", "model", "effort", "reply"):
            output.write(f"{key}<<{delimiter}\n{result[key]}\n{delimiter}\n")
        output.write(f"number={number}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as file:
            if result["decision"] == "run":
                file.write(f"Claude model: `{result['model']}` ({result['reason']})"
                           + (f", effort `{result['effort']}`" if result["effort"] else "") + "\n")
            else:
                file.write(f"Claude request refused: {result['reason']}\n")


if __name__ == "__main__":
    main()
