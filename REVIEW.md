# Review base branch

Empty base branch used as the target for a full-repository read-through.

Each `review/NN-*` branch adds one functional slice of the codebase on top of
this empty tree, so its pull request renders that slice as a pure-addition diff
that can be commented on line by line in the GitHub review UI.

- Snapshot of `main`: `83cfeb02abb89addf9d3c0aedc050489fc85b073`
- **These branches are for reading only. Do not merge any of them.**
