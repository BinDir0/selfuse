# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues in `BinDir0/RoWaH`. Use
the `gh` CLI for all operations.

Always pass `-R BinDir0/RoWaH` to `gh issue ...` commands because this checkout
has multiple GitHub remotes.

## Conventions

- **Create an issue**: `gh issue create -R BinDir0/RoWaH --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view -R BinDir0/RoWaH <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list -R BinDir0/RoWaH --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment -R BinDir0/RoWaH <number> --body "..."`
- **Apply / remove labels**: `gh issue edit -R BinDir0/RoWaH <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close -R BinDir0/RoWaH <number> --comment "..."`

Do not infer the repo from `git remote -v` for issue operations in this
checkout.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view -R BinDir0/RoWaH <number> --comments`.
