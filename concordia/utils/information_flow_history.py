"""Information flow history bank for tracking all LLM interactions."""

import datetime
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from typing_extensions import override


@dataclass
class ModelInteraction:
    """Single model interaction record."""
    timestamp: datetime.datetime
    agent_name: str
    component_name: str | None  # Which component made the call
    method: str  # 'sample_text' or 'sample_choice'
    prompt: str  # Full prompt sent to model
    response: str  # Model response
    kwargs: dict[str, Any]  # Model call parameters (temperature, max_tokens, etc.)
    metadata: dict[str, Any]  # Additional context (choice index, etc.)
    turn: int | None = None  # Simulation turn number if available
    phase: str | None = None  # 'pre_act', 'act', 'post_act', 'observe'


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


class InformationFlowHistoryBank:
    """Stores complete information flow history for all agents."""

    def __init__(self, save_dir: str | None = None):
        """Initialize history bank.

        Args:
            save_dir: Directory where history will be saved (optional).
        """
        self._interactions: dict[str, list[ModelInteraction]] = {}  # agent_name -> list
        self._lock = threading.Lock()
        self._save_dir = save_dir
        self._turn_counter: dict[str, int] = {}  # agent_name -> current turn

    def add_interaction(
        self,
        agent_name: str,
        prompt: str,
        response: str,
        method: str,
        kwargs: dict[str, Any] | None = None,
        component_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        phase: str | None = None,
    ) -> None:
        """Add a model interaction to the history.

        Args:
            agent_name: Name of the agent making the call.
            prompt: Full prompt sent to the model.
            response: Model response.
            method: Method name ('sample_text' or 'sample_choice').
            kwargs: Model call parameters (temperature, max_tokens, etc.).
            component_name: Name of component making the call (optional).
            metadata: Additional metadata (choice index, etc.).
            phase: Simulation phase ('pre_act', 'act', 'post_act', 'observe').
        """
        with self._lock:
            if agent_name not in self._interactions:
                self._interactions[agent_name] = []
                self._turn_counter[agent_name] = 0

            interaction = ModelInteraction(
                timestamp=datetime.datetime.now(),
                agent_name=agent_name,
                component_name=component_name,
                method=method,
                prompt=prompt,
                response=response,
                kwargs=kwargs or {},
                metadata=metadata or {},
                turn=self._turn_counter[agent_name],
                phase=phase,
            )
            self._interactions[agent_name].append(interaction)

    def get_agent_history(self, agent_name: str) -> list[ModelInteraction]:
        """Get all interactions for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            List of ModelInteraction records for the agent.
        """
        with self._lock:
            return list(self._interactions.get(agent_name, []))

    def get_all_history(self) -> dict[str, list[ModelInteraction]]:
        """Get all interactions for all agents.

        Returns:
            Dictionary mapping agent names to their interaction lists.
        """
        with self._lock:
            return {
                name: list(interactions)
                for name, interactions in self._interactions.items()
            }

    def increment_turn(self, agent_name: str) -> None:
        """Increment turn counter for an agent.

        Args:
            agent_name: Name of the agent.
        """
        with self._lock:
            if agent_name not in self._turn_counter:
                self._turn_counter[agent_name] = 0
            self._turn_counter[agent_name] += 1

    def get_current_turn(self, agent_name: str) -> int:
        """Get current turn number for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Current turn number (0-indexed, so first turn is 0).
        """
        with self._lock:
            return self._turn_counter.get(agent_name, 0)

    def save_to_json(self, filepath: str | None = None) -> str:
        """Save history to JSON file.

        Args:
            filepath: Path to save file. If None, generates filename in save_dir.

        Returns:
            Path to saved file.

        Raises:
            ValueError: If filepath is None and save_dir is None.
        """
        if filepath is None:
            if self._save_dir is None:
                raise ValueError("No save directory specified and no filepath provided")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self._save_dir, f"information_flow_history_{timestamp}.json"
            )

        with self._lock:
            data = {
                agent_name: [
                    asdict(interaction) for interaction in interactions
                ]
                for agent_name, interactions in self._interactions.items()
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=DateTimeEncoder, ensure_ascii=False)

        return filepath

    def load_from_json(self, filepath: str) -> None:
        """Load history from JSON file.

        Args:
            filepath: Path to JSON file to load.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with self._lock:
            for agent_name, interactions_data in data.items():
                interactions = []
                for item in interactions_data:
                    # Convert timestamp string back to datetime
                    if isinstance(item.get('timestamp'), str):
                        item['timestamp'] = datetime.datetime.fromisoformat(
                            item['timestamp']
                        )
                    interactions.append(ModelInteraction(**item))
                self._interactions[agent_name] = interactions

                # Set turn counter to max turn + 1
                if interactions:
                    max_turn = max(
                        (i.turn for i in interactions if i.turn is not None),
                        default=-1
                    )
                    self._turn_counter[agent_name] = max_turn + 1
                else:
                    self._turn_counter[agent_name] = 0

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text while preserving readability.

        Args:
            text: Text to truncate.
            max_length: Maximum length (0 = no limit).

        Returns:
            Truncated text with '...' if needed.
        """
        if max_length == 0 or len(text) <= max_length:
            return text

        # Try to truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If space is reasonably close
            truncated = truncated[:last_space]

        return truncated + "..."

    def _format_compact(
        self,
        interaction: ModelInteraction,
        max_input: int,
        max_output: int
    ) -> str:
        """Format interaction in compact format.

        Args:
            interaction: The interaction to format.
            max_input: Maximum input length (0 = no limit).
            max_output: Maximum output length (0 = no limit).

        Returns:
            Formatted string.
        """
        receiver = interaction.agent_name
        input_text = self._truncate_text(interaction.prompt, max_input)
        output_text = self._truncate_text(interaction.response, max_output)

        # Replace newlines with spaces for compact format (keep it on one line)
        input_text = input_text.replace('\n', ' ').replace('\r', ' ')
        output_text = output_text.replace('\n', ' ').replace('\r', ' ')

        # Escape quotes in text to avoid breaking the format
        input_text = input_text.replace('"', '\\"')
        output_text = output_text.replace('"', '\\"')

        turn_str = f"Turn {interaction.turn}" if interaction.turn is not None else "Turn ?"
        return f"[{turn_str}] [receiver: {receiver}] [input: \"{input_text}\"] [output: \"{output_text}\"]"

    def _group_by_turn(
        self,
        interactions: list[ModelInteraction]
    ) -> dict[int, list[ModelInteraction]]:
        """Group interactions by turn number.

        Args:
            interactions: List of interactions to group.

        Returns:
            Dictionary mapping turn numbers to interaction lists.
        """
        grouped: dict[int, list[ModelInteraction]] = {}
        for interaction in interactions:
            turn = interaction.turn if interaction.turn is not None else -1
            if turn not in grouped:
                grouped[turn] = []
            grouped[turn].append(interaction)
        return grouped

    def generate_simplified_log(
        self,
        format: str = 'compact',  # 'compact', 'markdown', 'text'
        max_input_length: int = 0,  # 0 = no limit
        max_output_length: int = 0,  # 0 = no limit
        group_by_turn: bool = True,
    ) -> str:
        """Generate a simplified, human-readable log.

        Args:
            format: Output format ('compact', 'markdown', 'text').
            max_input_length: Maximum characters for input (0 = no limit).
            max_output_length: Maximum characters for output (0 = no limit).
            group_by_turn: Whether to group interactions by turn.

        Returns:
            Formatted log string.
        """
        with self._lock:
            # Collect all interactions
            all_interactions: list[ModelInteraction] = []
            for interactions in self._interactions.values():
                all_interactions.extend(interactions)

            # Sort by timestamp
            all_interactions.sort(key=lambda x: x.timestamp)

            # Count interactions per agent
            agent_counts: dict[str, int] = {}
            for agent_name, interactions in self._interactions.items():
                agent_counts[agent_name] = len(interactions)

            total_count = len(all_interactions)

            # Generate header
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"=== Information Flow Log ===\n"
            header += f"Date: {timestamp_str}\n"
            header += f"Total Interactions: {total_count}\n"
            if agent_counts:
                agent_list = ", ".join(f"{name} ({count})" for name, count in agent_counts.items())
                header += f"Agents: {agent_list}\n"
            header += "\n"

            if format == 'compact':
                # Compact format: one line per interaction
                lines = [header]
                if group_by_turn:
                    grouped = self._group_by_turn(all_interactions)
                    for turn in sorted(grouped.keys()):
                        for interaction in grouped[turn]:
                            lines.append(self._format_compact(interaction, max_input_length, max_output_length))
                else:
                    for interaction in all_interactions:
                        lines.append(self._format_compact(interaction, max_input_length, max_output_length))

                return '\n'.join(lines)

            elif format == 'markdown':
                # Markdown format: structured with headers
                lines = [f"# Information Flow Log\n", f"**Date**: {timestamp_str}  \n", f"**Total Interactions**: {total_count}\n"]
                if agent_counts:
                    agent_list = ", ".join(f"{name} ({count})" for name, count in agent_counts.items())
                    lines.append(f"**Agents**: {agent_list}\n")
                lines.append("\n")

                if group_by_turn:
                    grouped = self._group_by_turn(all_interactions)
                    for turn in sorted(grouped.keys()):
                        lines.append(f"## Turn {turn}\n\n")
                        for idx, interaction in enumerate(grouped[turn], 1):
                            lines.append(f"### Interaction {idx}\n")
                            lines.append(f"- **Receiver**: {interaction.agent_name}\n")
                            if interaction.component_name:
                                lines.append(f"- **Component**: {interaction.component_name}\n")
                            if interaction.phase:
                                lines.append(f"- **Phase**: {interaction.phase}\n")

                            input_text = self._truncate_text(interaction.prompt, max_input_length)
                            output_text = self._truncate_text(interaction.response, max_output_length)

                            lines.append(f"- **Input**: \n  ```\n  {input_text}\n  ```\n")
                            lines.append(f"- **Output**: \n  ```\n  {output_text}\n  ```\n")
                            lines.append("\n")
                else:
                    for idx, interaction in enumerate(all_interactions, 1):
                        lines.append(f"### Interaction {idx}\n")
                        lines.append(f"- **Receiver**: {interaction.agent_name}\n")
                        if interaction.component_name:
                            lines.append(f"- **Component**: {interaction.component_name}\n")
                        if interaction.phase:
                            lines.append(f"- **Phase**: {interaction.phase}\n")

                        input_text = self._truncate_text(interaction.prompt, max_input_length)
                        output_text = self._truncate_text(interaction.response, max_output_length)

                        lines.append(f"- **Input**: \n  ```\n  {input_text}\n  ```\n")
                        lines.append(f"- **Output**: \n  ```\n  {output_text}\n  ```\n")
                        lines.append("\n")

                return ''.join(lines)

            elif format == 'text':
                # Text format: simple indentation
                lines = [header]
                if group_by_turn:
                    grouped = self._group_by_turn(all_interactions)
                    for turn in sorted(grouped.keys()):
                        lines.append(f"Turn {turn}:")
                        for idx, interaction in enumerate(grouped[turn], 1):
                            lines.append(f"  Interaction {idx}:")
                            lines.append(f"    Receiver: {interaction.agent_name}")
                            if interaction.component_name:
                                lines.append(f"    Component: {interaction.component_name}")
                            if interaction.phase:
                                lines.append(f"    Phase: {interaction.phase}")

                            input_text = self._truncate_text(interaction.prompt, max_input_length)
                            output_text = self._truncate_text(interaction.response, max_output_length)

                            lines.append(f"    Input: {input_text}")
                            lines.append(f"    Output: {output_text}")
                            lines.append("")
                else:
                    for idx, interaction in enumerate(all_interactions, 1):
                        lines.append(f"Interaction {idx}:")
                        lines.append(f"  Receiver: {interaction.agent_name}")
                        if interaction.component_name:
                            lines.append(f"  Component: {interaction.component_name}")
                        if interaction.phase:
                            lines.append(f"  Phase: {interaction.phase}")

                        input_text = self._truncate_text(interaction.prompt, max_input_length)
                        output_text = self._truncate_text(interaction.response, max_output_length)

                        lines.append(f"  Input: {input_text}")
                        lines.append(f"  Output: {output_text}")
                        lines.append("")

                return '\n'.join(lines)

            else:
                raise ValueError(f"Unknown format: {format}. Must be 'compact', 'markdown', or 'text'")

    def save_simplified_log(
        self,
        filepath: str | None = None,
        format: str = 'compact',
        max_input_length: int = 0,
        max_output_length: int = 0,
    ) -> str:
        """Save simplified log to file.

        Args:
            filepath: Path to save file. If None, generates filename in save_dir.
            format: Output format ('compact', 'markdown', 'text').
            max_input_length: Maximum characters for input (0 = no limit).
            max_output_length: Maximum characters for output (0 = no limit).

        Returns:
            Path to saved file.

        Raises:
            ValueError: If filepath is None and save_dir is None.
        """
        if filepath is None:
            if self._save_dir is None:
                raise ValueError("No save directory specified and no filepath provided")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = '.md' if format == 'markdown' else '.txt'
            filepath = os.path.join(
                self._save_dir, f"information_flow_simplified_{timestamp}{extension}"
            )

        log_content = self.generate_simplified_log(
            format=format,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            group_by_turn=True,
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(log_content)

        return filepath
