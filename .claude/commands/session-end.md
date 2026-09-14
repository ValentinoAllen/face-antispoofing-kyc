---
description: Close the working session. Update PROGRESS.md and EXPERIMENTS.md from what actually happened, then commit and push.
argument-hint: "[optional note about the session]"
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*), Bash(git rev-parse:*), Bash(git add:*), Bash(git commit:*), Bash(git push), Bash(date:*), Read, Edit, Write, Glob, Grep
---

End the current working session. Optional note from the owner: $ARGUMENTS

## Ground rules (read before doing anything)

- **Record only what actually happened in this session.**
  - Sources of truth: this conversation, `git status`, `git diff`, `git log`, and files that exist on
    disk.
  - Do not record intended, planned or "almost done" work as done.
  - A command that was not run did not pass.
- **Never invent numbers.**
  - Metrics, latencies, counts and sizes may only be copied from a run record
    (`reports/runs/<run_id>/record.json`) or from command output seen in this session.
  - If a value is not available, leave the cell as `—` or write `TBD — decide before <milestone>`.
- **Do not read or modify anything under `data/`.**
- **No attribution trailers.**
  - The commit message must not contain `Co-Authored-By` for any AI, a "Generated with" footer, a
    `Claude-Session:` line, or any mention of Claude, Anthropic or AI assistance.
  - Never use `--no-verify`.
- **Pushing** (`RULES.md` §6 item 5).
  - Push only in step 6, with a plain `git push`. Never use `--force`, `--force-with-lease`, `-f`
    or any other flag.
  - Do not add remotes.
  - Never report a push as successful unless `git push` returned success.

## Steps

### 1. Gather facts
- Get today's date: `date +%Y-%m-%d`.
- Run `git status --short`, `git diff --stat` and `git diff --cached --stat`.
- Read `docs/PROGRESS.md`. Find the date of the most recent session-log entry, then run
  `git log --since=<that date> --oneline`.
- List `reports/runs/*/record.json` files created or modified this session, if any.
- Look back over this conversation for what was attempted, what worked, and what failed.

If there are no file changes, no new commits and no runs, tell the owner there is nothing to record.
Ask whether they still want a log entry. Stop there if the answer is no.

### 2. Rewrite `## Current status` in `docs/PROGRESS.md`
Replace the existing paragraph **in place**. Do not append to it. Write one paragraph describing the
state of the project as it stands now: what works, what exists but is unverified, and the immediate
next step. Keep it factual and present tense.

### 3. Prepend a session-log entry
Insert a new entry directly under `## Session log`, above all older entries, in this format:

```
### YYYY-MM-DD: <short title of the session's main work>

**Done**
- <concrete things completed, with file paths>

**Broke / not verified**
- <failures, errors, flaky tests, things built but not run. Write "Nothing observed" only if that is true>

**Next**
- <the next concrete actions>
```

Do not edit older entries. Also update, as far as this session supports it:
- `## Milestones`: tick only items that are fully done; add the completion date.
- `## Open questions`: add new ones, and strike through resolved ones with the answer.
- `## Blocked on`: add or remove items.

### 4. Append completed runs to `docs/EXPERIMENTS.md`
For each run with `status: "completed"` that is not yet in the table:
- Add a row at the **bottom** of the table, with values copied from its record.
- Take the hypothesis and conclusion from the record and this conversation.
- If a record is missing a value, use `—`.

Never add rows for failed, aborted or running runs; mention those in the session log instead. Never
modify existing rows.

### 5. Stage and commit
1. `git add -A`
2. Run `git diff --cached --name-only` and inspect the list. **Abort and tell the owner** if it
   includes anything under `data/`, `wandb/`, a `*.pt`/`*.pth`/`*.ckpt`/`*.onnx` file, a `.env*`
   file, or anything that looks like a credential.
3. Commit with a plain imperative message written as the repository owner, for example:
   `Add subject-disjoint split builder and update progress log`. Subject line of 72 characters or
   fewer; an optional short body listing the main changes.
4. Run `git log -1 --format=full` and show the output. Confirm there are no trailers.

### 6. Push
Run this step only if step 5 created a commit.
1. Run `git push` exactly, with no flags and no arguments. Never `--force` or `--force-with-lease`.
2. If it returns success, keep the output line that shows the updated ref.
3. If it fails for any reason (no upstream, rejected, authentication, network, hook), do not retry
   with other flags, and do not pull, rebase or amend to make it succeed. Show the actual error
   output and state plainly: **the commit is local only; it was not pushed.**

### 7. Report back
Give a short summary: the new Current status paragraph, the session-log entry, any ledger rows
added, the commit SHA, and the push result: either pushed (with the updated ref line) or local only
(with the error).
