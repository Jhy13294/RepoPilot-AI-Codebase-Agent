# Internal Templates (public copies)

The live `tasks/` and `memory/` directories are **gitignored** working files. These templates are
their public, stable counterparts so the methodology in `docs/project-management.md` is fully
reproducible by anyone cloning this repo.

| Template | Mirrors | Purpose |
|---|---|---|
| [`tasks-templates.md`](tasks-templates.md) | `tasks/todo.md`, `tasks/in-progress.md`, `tasks/done.md` | Kanban-style task board with construction IDs and acceptance criteria |
| [`memory-templates.md`](memory-templates.md) | `memory/project-context.md`, `memory/decisions.md`, `memory/lessons-learned.md` | Session-resumable context, ADR-style decision log, lessons log |

Usage: copy the blocks into `tasks/` and `memory/` at project start; both directories are already
listed in `.gitignore`.
