"""DAgger-style teacher gating over a student policy."""

from __future__ import annotations

import numpy as np


class DAggerPolicyWrapper:
    """Runs the student unless it strays too far from the teacher.

    Every act() queries both policies; with ``err = ||student - teacher||_2``,
    the student's action is returned while ``err < thresh`` and the teacher's
    once ``err >= thresh``. The per-step labels a DAgger data collector needs
    are kept on the wrapper: ``last_student_action`` / ``last_teacher_action``
    / ``last_err`` / ``last_intervened``, plus running ``steps`` and
    ``interventions`` counters (reset() zeroes them).
    """

    def __init__(self, student, teacher, thresh: float):
        self.student = student
        self.teacher = teacher
        self.thresh = thresh
        self.last_student_action: np.ndarray | None = None
        self.last_teacher_action: np.ndarray | None = None
        self.last_err: float = 0.0
        self.last_intervened: bool = False
        self.steps: int = 0
        self.interventions: int = 0

    def reset(self, obs=None) -> None:
        self.student.reset(obs)
        self.teacher.reset(obs)
        self.last_student_action = None
        self.last_teacher_action = None
        self.last_err = 0.0
        self.last_intervened = False
        self.steps = 0
        self.interventions = 0

    def act(self, obs=None) -> np.ndarray:
        student_action = np.asarray(self.student.act(obs))
        teacher_action = np.asarray(self.teacher.act(obs))
        err = float(np.linalg.norm(student_action - teacher_action))
        intervened = err >= self.thresh
        self.last_student_action = student_action
        self.last_teacher_action = teacher_action
        self.last_err = err
        self.last_intervened = intervened
        self.steps += 1
        self.interventions += int(intervened)
        return teacher_action if intervened else student_action
