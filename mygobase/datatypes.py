from dataclasses import dataclass


@dataclass
class Storyboard:
    id: int
    episode: int
    frame_number: int
    subtitle: str
    picture: bytes
    role: str | None
