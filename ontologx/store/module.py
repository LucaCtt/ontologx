from abc import ABC, abstractmethod


class StoreModule(ABC):
    """Abstract class for store modules."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store module by creating module-specific nodes."""
