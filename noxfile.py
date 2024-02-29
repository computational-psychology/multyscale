import nox


@nox.session(python="3.12")
def lint(session):
    # Run the linters (via pre-commit)
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)
