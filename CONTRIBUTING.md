# Contributing

The following is a set of guidelines for contributing to GraphSPME.

## Ground Rules

1. We use Black code formatting
1. All code must be testable and unit tested.

## Commits

We strive to keep a consistent and clean git history and all contributions should adhere to the following:

1. All tests should pass on all commits(*)
1. A commit should do one atomic change on the repository
1. The commit message should be descriptive.



We expect commit messages to follow this style:

1. Separate subject from body with a blank line
1. Limit the subject line to 50 characters
1. Capitalize the subject line
1. Do not end the subject line with a period
1. Use the imperative mood in the subject line
1. Wrap the body at 72 characters
1. Use the body to explain what and why vs. how

This list is taken from [here](https://chris.beams.io/posts/git-commit/).

Also, focus on making clear the reasons why you made the change in the first
place—the way things worked before the change (and what was wrong with that),
the way they work now, and why you decided to solve it the way you did. A
commit body is required for anything except very small changes.

(*) Tip for making sure all tests pass, try out --exec while rebasing. You
can then have all tests run per commit in a single command.

## Pull Request Scoping

Ideally a pull request will be small in scope, and atomic, addressing precisely
one issue, and mostly result in a single commit. It is however permissible to
fix minor details (formatting, linting, moving, simple refactoring ...) in the
vicinity of your work.

If you find that you want to do lots of changes that are not directly related
to the issue you're working on, create a seperate PR for them so as to avoid
noise in the review process.
