"""
Validates detected wire connections against a predefined specification.
"""
import json
from logger_setup import logger

logger.info("connection_validator.py loaded")

class ConnectionValidator:
    """
    Validates a list of detected wires against a JSON specification file.
    """
    def __init__(self, spec_file_path):
        """
        Initializes the ConnectionValidator.

        Args:
            spec_file_path (str): Path to the JSON file listing required connections.
                                  Each entry: {"from": "T1A", "to": "T2B", "color": "red"}
        """
        logger.info(f"Initializing ConnectionValidator with spec file: {spec_file_path}")
        self.spec_file_path = spec_file_path
        self.required_connections_spec = []
        try:
            with open(spec_file_path, 'r') as f:
                self.required_connections_spec = json.load(f)
            logger.info(f"Loaded {len(self.required_connections_spec)} required connections from spec.")
        except FileNotFoundError:
            logger.error(f"Specification file not found: {spec_file_path}")
            # Allow initialization, validation will just show all as missing
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from specification file: {spec_file_path}")
        except Exception as e:
            logger.error(f"Error loading specification file: {e}")

    def validate_connections(self, detected_wires_with_terminals):
        """
        Validates detected wires against the loaded specification.

        Args:
            detected_wires_with_terminals (list[Wire]): A list of Wire objects,
                where terminal_A and terminal_B have been populated.

        Returns:
            list[tuple]: A list of tuples (spec_entry, status_message).
                         Status can be "OK", "Missing", "Mismatched Color", etc.
        """
        logger.info(f"Starting connection validation for {len(detected_wires_with_terminals)} detected wires.")
        if not self.required_connections_spec:
            logger.warning("No connection specification loaded. Cannot validate.")
            return []

        results = []
        observed_connections = set()

        for wire in detected_wires_with_terminals:
            if wire.terminal_A and wire.terminal_B and wire.terminal_A != "N/A" and wire.terminal_B != "N/A":
                # Add both orderings for easier lookup, along with color
                observed_connections.add(tuple(sorted((wire.terminal_A, wire.terminal_B))) + (wire.color,))
        
        logger.debug(f"Observed connections (normalized): {observed_connections}")

        for spec_entry in self.required_connections_spec:
            spec_from = spec_entry.get("from")
            spec_to = spec_entry.get("to")
            spec_color = spec_entry.get("color")

            # Normalize spec connection for lookup (sorted terminals)
            normalized_spec_conn = tuple(sorted((spec_from, spec_to))) + (spec_color,)

            status_to_report = f"Missing ({spec_color})" # Default to missing
            found_spec_terminals_with_wrong_color = False

            if normalized_spec_conn in observed_connections:
                status_to_report = "OK"
                logger.debug(f"Spec entry {spec_entry} found: OK")
            else:
                # Check if the terminals are connected but with a different color wire
                spec_terminals_sorted = tuple(sorted((spec_from, spec_to)))
                for obs_conn_term_A, obs_conn_term_B, obs_color in observed_connections:
                    if spec_terminals_sorted == tuple(sorted((obs_conn_term_A, obs_conn_term_B))):
                        # Terminals match, but color was different (otherwise it would have been 'OK')
                        status_to_report = f"Mismatched Color (Found {obs_color}, Expected {spec_color})"
                        found_spec_terminals_with_wrong_color = True
                        break
                logger.warning(f"Spec entry {spec_entry} result: {status_to_report}")
            results.append((spec_entry, status_to_report))
        
        logger.info(f"Connection validation complete. Results count: {len(results)}")
        return results