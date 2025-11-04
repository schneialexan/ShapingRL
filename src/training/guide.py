from collections import deque
from copy import copy
from typing import Self


class GuidelineItem:
    def __init__(self, name: str, grid: int, cell: int, row: int, n_times: int):
        self.name = name
        self.grid = grid
        self.cell = cell
        self.row = row
        self.n_times = n_times
        self._visit_count = 0

    def visit(self):
        self._visit_count += 1

    @property
    def is_completed(self) -> bool:
        return self._visit_count >= self.n_times

    @property
    def n_times_visited(self) -> float:
        return self._visit_count


class Guideline:
    def __init__(self, guideline_items: deque[GuidelineItem]):
        self._guideline_items = guideline_items

    def get_guide_items(self) -> deque[GuidelineItem]:
        return copy(self._guideline_items)


class GuideLineBuilder:

    def __init__(self):
        self._checkpoints: deque[GuidelineItem] = deque()

    def set_goal_go_to(
        self,
        name: str,
        grid: int,
        row: int,
        cell: int,
        n_times: int = 1
    ) -> Self:
        self._checkpoints.append(
            GuidelineItem(name=name, n_times=n_times, grid=grid, cell=cell, row=row)
        )

        return self

    def then_set_goal_go_to(
        self,
        name: str,
        grid: int,
        row: int,
        cell: int,
        n_times: int = 1
    ) -> Self:
        self.set_goal_go_to(
            name=name,
            n_times=n_times,
            grid=grid,
            cell=cell,
            row=row
        )

        return self

    def build(self) -> Guideline:
        guideline_items = copy(self._checkpoints)

        return Guideline(guideline_items)
