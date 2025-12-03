#!/usr/bin/env python3
"""
Two LLM agents conversing with PE-driven adaptation (OpenAI API version)
-------------------------------------------------------------------------
- Each agent keeps memory: conversation, PE (prediction error) per turn, reflections, and goal.
- Turn loop: ACT (speaker) -> OBSERVE (listener estimates state & computes PE) -> LEARNING (listener reflects).
- Logs per-turn details and saves to JSON on completion.

Quickstart
----------
1) pip install -U openai
2) export OPENAI_API_KEY="sk-..."
3) python pe_conversation_openai.py --turns 6 --model gpt-4o-mini

Notes
-----
- Uses the Responses API (client.responses.create). Adjust model as needed.
- Set --temperature, --top_p for sampling. Defaults are conservative to reduce variance.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# -------------------------
# OpenAI client
# -------------------------
try:
    from openai import OpenAI
except Exception as e:
    print("ERROR: Failed to import OpenAI client. Install with: pip install -U openai", file=sys.stderr)
    raise

# LLM callable type
LLMFn = Callable[[str], str]

# API key
os.environ["OPENAI_API_KEY"] = "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"

def make_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.2, top_p: float = 0.9, max_retries: int = 3, timeout_s: float = 30.0) -> LLMFn:
    """
    Returns a callable llm(prompt:str)->str using OpenAI Responses API.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # reads OPENAI_API_KEY from env

    def llm(prompt: str) -> str:
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    # You can set max_output_tokens if you want strict caps, e.g. 256
                )
                # responses.create returns a rich object; output_text gives best-effort text
                text = getattr(resp, "output_text", None)
                if text is None:
                    # Fallback: stitch together text parts if output_text missing
                    text = ""
                    if getattr(resp, "output", None):
                        for item in resp.output:
                            if getattr(item, "content", None):
                                for part in item.content:
                                    if getattr(part, "type", "") == "output_text" and getattr(part, "text", None):
                                        text += part.text
                return (text or "").strip()
            except Exception as e:
                last_exc = e
                # basic exponential backoff
                wait = min(2 ** (attempt - 1), 8)
                time.sleep(wait)
        # If we exhausted retries, raise the last exception
        raise last_exc

    return llm

# -------------------------
# Data structures
# -------------------------

@dataclass
class Goal:
    name: str                 # e.g., "likability"
    description: str          # definition/instructions
    ideal: float = 1.0        # numeric target (1.0 == 100%)

@dataclass
class PERecord:
    turn: int
    partner_text: str
    estimate: float           # estimated current state in [0,1]
    pe: float                 # ideal - estimate

@dataclass
class ReflectionRecord:
    turn: int
    text: str

@dataclass
class Utterance:
    turn: int
    speaker: str
    text: str

@dataclass
class AgentMemory:
    goal: Goal
    conversation: List[Utterance] = field(default_factory=list)
    pe_history: List[PERecord] = field(default_factory=list)
    reflections: List[ReflectionRecord] = field(default_factory=list)

    def last_k(self, k: int) -> Tuple[List[Utterance], List[PERecord], List[ReflectionRecord]]:
        return (self.conversation[-k:], self.pe_history[-k:], self.reflections[-k:])

# -------------------------
# Agent
# -------------------------

class Agent:
    def __init__(self, name: str, goal: Goal, llm: LLMFn, recent_k: int = 3, seed: int = 0):
        self.name = name
        self.llm = llm
        self.memory = AgentMemory(goal=goal)
        self.recent_k = recent_k
        self._rng = random.Random(seed)

    # ---------- Observe ----------
    def observe(self, turn: int, partner_text: str) -> PERecord:
        """
        Prompt LLM: estimate current state on the goal dimension in [0,1] from partner's last response.
        Compute PE = ideal - estimate. Store and return PERecord.
        """
        prompt = f"""You are {self.name}. Goal: {self.memory.goal.name}.
        Goal description: {self.memory.goal.description}
        Ideal value on goal dimension: {self.memory.goal.ideal:.2f}

        Task: From only the partner's last response, estimate the CURRENT STATE on the goal dimension
        as a single number in [0,1], where 1 means perfectly achieving the goal.
        Partner said: "{partner_text}"

        Respond with a single number in [0,1]. You may include a brief comment after the number.
        """
        raw = self.llm(prompt)
        # Parse first float in [0,1]
        m = re.search(r"([01](?:\.\d+)?|\d\.\d+)", raw)
        estimate = float(m.group(1)) if m else 0.5
        estimate = max(0.0, min(1.0, estimate))
        pe = self.memory.goal.ideal - estimate
        rec = PERecord(turn=turn, partner_text=partner_text, estimate=estimate, pe=pe)
        self.memory.pe_history.append(rec)
        return rec

    # ---------- Learning ----------
    def learning(self, turn: int) -> ReflectionRecord:
        """
        Retrieve most recent PE and ask LLM for a reflection to reduce PE next turn.
        """
        pe_last = self.memory.pe_history[-1].pe if self.memory.pe_history else 0.0
        prompt = f"""You are {self.name}. Goal: {self.memory.goal.name}.
        Given the PE of last turn (PE_last = {pe_last:+.3f}), write a short reflection:
        What will you change next turn to REDUCE PE? Keep it concrete and brief.
        """
        text = self.llm(prompt).strip()
        rec = ReflectionRecord(turn=turn, text=text)
        self.memory.reflections.append(rec)
        return rec

    # ---------- Act ----------
    def act(self, turn: int) -> str:
        """
        Produce an utterance conditioned on recent conversation, PEs, reflections, and the goal.
        """
        conv_k, pe_k, refl_k = self.memory.last_k(self.recent_k)

        def fmt_conv(u: Utterance) -> str:
            return f"[t={u.turn} {u.speaker}] {u.text}"

        def fmt_pe(p: PERecord) -> str:
            return f"(turn {p.turn}) estimate={p.estimate:.2f}, PE={p.pe:+.2f} ← partner: “{p.partner_text}”"

        def fmt_refl(r: ReflectionRecord) -> str:
            return f"(turn {r.turn}) {r.text}"

        prompt = f"""You are {self.name}. Your goal is "{self.memory.goal.name}".
        Definition: {self.memory.goal.description}
        Ideal value: {self.memory.goal.ideal:.2f}

        You must talk in a way that MINIMIZES PREDICTION ERROR (PE = ideal - estimated current state).
        Consider recent conversation, PE history, and your reflections.

        Recent conversation (last {self.recent_k}):
        {chr(10).join("- " + fmt_conv(u) for u in conv_k) or "- (none)"}

        Recent PE history:
        {chr(10).join("- " + fmt_pe(p) for p in pe_k) or "- (none)"}

        Recent reflections:
        {chr(10).join("- " + fmt_refl(r) for r in refl_k) or "- (none)"}

        Now produce ONE concise utterance to your partner that is likely to REDUCE PE next turn.
        Avoid meta-talk; speak naturally.
        """
        text = self.llm(prompt).strip()
        # Store own utterance
        self.memory.conversation.append(Utterance(turn=turn, speaker=self.name, text=text))
        return text

# -------------------------
# Orchestrator
# -------------------------

@dataclass
class TurnLog:
    time: str
    turn: int
    speaker: str
    listener: str
    speaker_text: str
    listener_estimate: float
    listener_pe: float
    listener_reflection: str

class ConversationStudy:
    def __init__(self, agent_a: Agent, agent_b: Agent, total_turns: int = 6, seed: int = 13):
        self.A = agent_a
        self.B = agent_b
        self.total_turns = total_turns
        self._rng = random.Random(seed)
        self.log: List[TurnLog] = []

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def run(self) -> List[TurnLog]:
        for t in range(1, self.total_turns + 1):
            speaker, listener = (self.A, self.B) if t % 2 == 1 else (self.B, self.A)

            # Speaker acts
            text = speaker.act(turn=t)

            # Share utterance with listener's conversation memory
            listener.memory.conversation.append(Utterance(turn=t, speaker=speaker.name, text=text))

            # Listener observes -> estimate + PE
            pe_rec = listener.observe(turn=t, partner_text=text)

            # Listener learning -> reflection
            refl = listener.learning(turn=t)

            # Log
            self.log.append(TurnLog(
                time=self._ts(),
                turn=t,
                speaker=speaker.name,
                listener=listener.name,
                speaker_text=text,
                listener_estimate=pe_rec.estimate,
                listener_pe=pe_rec.pe,
                listener_reflection=refl.text
            ))

        return self.log

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(l) for l in self.log], f, ensure_ascii=False, indent=2)

# -------------------------
# CLI / main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-agent PE conversation (OpenAI API).")
    parser.add_argument("--turns", type=int, default=2, help="Total turns (messages) in the dialogue.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name, e.g., gpt-4o-mini.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--window", type=int, default=3, help="Recent K turns to condition on.")
    parser.add_argument("--outfile", type=str, default="pe_conversation_log.json", help="Where to save JSON log.")
    args = parser.parse_args()

    # Ensure API key exists
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Export it before running.", file=sys.stderr)
        sys.exit(1)

    # Define the shared goal
    goal = Goal(
        name="likability",
        description="Be perceived as likable by the partner (0=not liked, 1=fully liked). Aim for 1.0.",
        ideal=1.0
    )

    # Build the LLM wrapper
    llm = make_openai_llm(model=args.model, temperature=args.temperature, top_p=args.top_p)

    # Create agents
    A = Agent(name="Agent A", goal=goal, llm=llm, recent_k=args.window, seed=1)
    B = Agent(name="Agent B", goal=goal, llm=llm, recent_k=args.window, seed=2)

    # Run
    study = ConversationStudy(A, B, total_turns=args.turns, seed=7)
    runlog = study.run()

    # Pretty print concise trace
    for r in runlog:
        print(f"[t={r.turn}] {r.speaker} → {r.listener}: {r.speaker_text}")
        print(f"       {r.listener} observed estimate={r.listener_estimate:.2f}, PE={r.listener_pe:+.2f}")
        print(f"       {r.listener} reflection: {r.listener_reflection}\n")

    # Save full JSON
    study.save_json(args.outfile)
    print(f"Saved detailed log → {args.outfile}")

if __name__ == "__main__":
    main()
