# Contributing to DeepSensor

🌍💫 We're excited that you're here and want to contribute. 💫🌍

By joining our efforts, you will be helping to push the frontiers of environmental sciences.

We want to ensure that every user and contributor feels welcome, included and supported to
participate in DeepSensor community. Whether you're a seasoned developer, a machine learning
researcher, an environmental scientist, or just someone eager to learn and contribute, **you are
welcome here**. We value every contribution, be it big or small, and we appreciate the unique
perspectives you bring to the project.

We hope that the information provided in this document will make it as easy as possible for you to
get involved. If you find that you have questions that are not discussed below, please let us know
through one of the many ways to [get in touch](#get-in-touch).

## Important Resources

If you'd like to find out more about DeepSensor, make sure to check out:

1. **README**: For a high-level overview of the project, please refer to our README.
2. **Documentation**: For more detailed information about the project, please refer to
   our [documentation](https://alan-turing-institute.github.io/deepsensor).
3. **Project Roadmap**: Familiarize yourself with our direction and goals by checking
   out [the project's roadmap](https://alan-turing-institute.github.io/deepsensor/community/roadmap.html).

## Get in touch

The easiest way to get involved with the active development of DeepSensor is to join our regular
community calls. The community calls are currently on a hiatus but if you are interested in
participating in the forthcoming community calls, which will start in 2024, you should join our
Slack workspace, where conversation about when to hold the community calls in the future will take
place.

**Slack Workspace**: Join
our DeepSensor Slack channel for
discussions, queries, and community interactions. In order to join, [sign up for the Turing Environment & Sustainability stakeholder community](https://forms.office.com/pages/responsepage.aspx?id=p_SVQ1XklU-Knx-672OE-ZmEJNLHTHVFkqQ97AaCfn9UMTZKT1IwTVhJRE82UjUzMVE2MThSOU5RMC4u). The form includes a question on signing up for the Slack team, where you can find DeepSensor's channel.

**Email**: If you prefer a more formal communication method or have specific concerns, please reach
us at tomandersson3@gmail.com.

## How to Contribute

We welcome contributions of all kinds, be it code, documentation, raising issues, or community engagement. We
encourage you to read through the following sections to learn more about how you can contribute to

### How to Submit Changes

We follow the same instructions for submitting changes to the project as those developed
by [The Turing Way](https://github.com/the-turing-way/the-turing-way/blob/main/CONTRIBUTING.md#making-a-change-with-a-pull-request).
In short, there are five steps to adding changes to this repository:

1. **Fork the Repository**: Start
   by [forking the DeepSensor repository](https://github.com/alan-turing-institute/deepsensor/fork).
2. **Make Changes**: Ensure your code adheres to the style guidelines and passes all tests.
3. **Commit and Push**: Use clear commit messages.
4. **Open a Pull Request**: Ensure you describe the changes made and any additional details.

#### 1. Fork the Repository

Once you have [created a fork of the repository](https://github.com/alan-turing-institute/deepsensor/fork),
you now have your own unique local copy of DeepSensor. Changes here won't affect anyone else's work,
so it's a safe space to explore edits to the code!

Make sure to [keep your fork up to date](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) with the main repository, otherwise, you
can end up with lots of dreaded [merge conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts).

If you prefer working with GitHub in the
browser, [these instructions](https://github.com/KirstieJane/STEMMRoleModels/wiki/Syncing-your-fork-to-the-original-repository-via-the-browser)
describe how to sync your fork to the original repository.

#### 2. Make Changes

Try to keep the changes focused.
If you submit a large amount of work all in one go it will be much more work for whoever is
reviewing your pull request.
Help them help you! :wink:

Are you new to Git and GitHub or just want a detailed guide on getting started with version control?
Check out
our [Version Control chapter](https://the-turing-way.netlify.com/version_control/version_control.html)
in _The Turing Way_ Book!

#### 3. Commit and Push

While making your changes, commit often and write good, detailed commit messages.
[This blog](https://chris.beams.io/posts/git-commit/) explains how to write a good Git commit
message and why it matters.
It is also perfectly fine to have a lot of commits - including ones that break code.
A good rule of thumb is to push up to GitHub when you _do_ have passing tests then the continuous
integration (CI) has a good chance of passing everything. 😸

Please do not re-write history!
That is, please do not use the [rebase](https://help.github.com/en/articles/about-git-rebase)
command to edit previous commit messages, combine multiple commits into one, or delete or revert
commits that are no longer necessary.

Make sure you're using the developer dependencies.
If you're working locally on the source code, *before* commiting, please run `pip install -r requirements/requirements.dev.txt` to install some useful dependencies just for development.
This includes `pre-commit` and `ruff` which are used to check and format your code style when you run `git commit`, so that you don't have to.
To make this work, just run `pre-commit install`.

You should also run `pytest` and check that your changes don't break any of the existing tests.
If you've made changes to the source code, you may need to add some tests to make sure that they don't get broken in the future.

#### 4. Open a Pull Request

We encourage you to open a pull request as early in your contributing process as possible.
This allows everyone to see what is currently being worked on.
It also provides you, the contributor, feedback in real-time from both the community and the
continuous integration as you make commits (which will help prevent stuff from breaking).

GitHub has a [nice introduction](https://guides.github.com/introduction/flow) to the pull request
workflow, but please [get in touch](#get-in-touch) if you have any questions :balloon:.

### DeepSensor's documentation

You don't have to write code to contribute to DeepSensor.
Another highly valuable way of contributing is helping with DeepSensor's [documentation](https://alan-turing-institute.github.io/deepsensor).
See below for information on how to do this.

#### Background

We use the Jupyter Book framework to build our documentation in the `docs/` folder.
The documentation is written in
Markdown and Jupyter Notebooks. The documentation is hosted on GitHub Pages and is automatically
built and deployed using GitHub Actions after every commit to the `main` branch.

DeepSensor requires slightly unique documentation, because demonstrating the package requires
both data and trained models.
This makes it compute- and data-hungry to run some of the notebooks, and they cannot
run on GitHub Actions.
Therefore, all the notebooks are run locally - the code cell outputs are saved in the .ipynb files
and are rendered when the documentation is built.
If DeepSensor is updated, some of the notebooks may become out of date and will need to be re-run.

Some relevant links for Juptyer Book and MyST:
* https://jupyterbook.org/en/stable/intro.html
* https://jupyterbook.org/en/stable/content/myst.html
* https://jupyterbook.org/en/stable/reference/cheatsheet.html

#### Contributing to documentation

One easy way to contribute to the documentation is to provide feedback in [this issue](https://github.com/alan-turing-institute/deepsensor/issues/87) and/or in the DeepSensor Slack channel.

Another way to contribute is to directly edit or add to the documentation and open a PR:
* Follow all the forking instructions above
* Install the documentation requirements: `pip install -r requirements/requirements.docs.txt`
* Option A: Editing a markdown file
  * Simply make your edits!
* Option B: Editing a jupyter notebook file
  * This can be more involved... Firstly, reach out on the Slack channel to ask if anyone else is working on the same notebook file locally. Working one-at-a-time can save Jupyter JSON merge conflict headaches later!
  * If you are only editing markdown cells, just re-run those cells specifically to compile them
  * If you are editing code cells:
    * Install `cartopy` using `conda install -c conda-forge cartopy`
    * Run the all the code cells that the current cell depends on and any subsequent code cells that depend on the current cell (you may need to rerun the whole notebook)
    * Note: Some notebooks require a GPU and some assume that previous notebooks have been run
  * Please be careful about not clearing any code cell outputs that you don't intend to!
* Once your changes have been made, regenerate the docs locally with `jupyter-book build docs --all` and check your changes have applied as expected
* Push your changes and open a PR (see above)

## First-timers' Corner

If you're new to the project, we recommend starting with issues labeled
as ["good first issue"](https://github.com/alan-turing-institute/deepsensor/issues?q=is:issue+is:open+label:%22good+first+issue%22).
These are typically simpler tasks that offer a great starting point. Browse these here.

There's also the
label ["thoughts welcome"](https://github.com/alan-turing-institute/deepsensor/issues?q=is:issue+is:open+label:%22thoughts+welcome%22),
which allows for you to contribute with discussion points in the issues, even if you don't want to
or cannot contribute to the codebase.

If you feel ready for it, you can also open a new issue. Before you open a new issue, please check
if any of [our open issues](https://github.com/alan-turing-institute/deepsensor/issues) cover your idea
already. If you open a new issue, please follow our basic guidelines laid out in our issue
templates, which you should be able to see if
you [open a new issue](https://github.com/alan-turing-institute/deepsensor/issues/new/choose).

## Reporting Bugs

Found a bug? Please open an issue here on GitHub to report it. We have a template for opening
issues, so make sure you follow the correct format and ensure you include:

- A clear title.
- A detailed description of the bug.
- Steps to reproduce it.
- Expected versus actual behavior.

## Recognising Contributions

We value and recognize every contribution. All contributors will be acknowledged in the
[contributors](https://github.com/alan-turing-institute/deepsensor/tree/main#contributors) section of the
README.
Notable contributions will also be highlighted in our fortnightly community meetings.

DeepSensor follows the [all-contributors](https://github.com/kentcdodds/all-contributors#emoji-key)
specifications. The all-contributors bot usage is
described [here](https://allcontributors.org/docs/en/bot/usage). You can see a list of current
contributors here.

To add yourself or someone else as a contributor, comment on the relevant Issue or Pull Request with
the following:

> @all-contributors please add username for contribution1, contribution2

You can see
the [Emoji Key (Contribution Types Reference)](https://allcontributors.org/docs/en/emoji-key) for a
list of valid <contribution> types and examples of how this command can be run
in [this issue](https://github.com/alan-turing-institute/deepsensor/issues/58). The bot will then create a
Pull Request to add the contributor and reply with the pull request details.

**PLEASE NOTE: Only one contributor can be added with the bot at a time!** Add each contributor in
turn, merge the pull request and delete the branch (`all-contributors/add-<username>`) before adding
another one. Otherwise, you can end up with
dreaded [merge conflicts](https://help.github.com/articles/about-merge-conflicts). Therefore, please
check the open pull requests first to make sure there aren't
any [open requests from the bot](https://github.com/alan-turing-institute/deepsensor/pulls/app%2Fallcontributors)
before adding another.

What happens if you accidentally run the bot before the previous run was merged and you got those
pesky merge conflicts? (Don't feel bad, we have all done it! 🙈) Simply close the pull request and
delete the branch (`all-contributors/add-<username>`). If you are unable to do this for any reason,
please let us know on Slack <link to Slack> or by opening an issue, and one of our core team members
will be very happy to help!

## Need Help?

If you're stuck or need assistance:

- Check our [FAQ](https://alan-turing-institute.github.io/deepsensor/community/faq.html) section first.
- Reach out on Slack or via email for personalized assistance. (See ["Get in touch"](#get-in-touch)
  above for links.)
- Consider pairing up with a another contributor for guidance. You can always find us in the Slack
  channel and we're happy to chat!

**Once again, thank you for considering contributing to DeepSensor! We hope you enjoy your
contributing experience.**

## Inclusivity

We aim to make DeepSensor a collaboratively developed project. We, therefore, require that all our
members and their contributions **adhere to our [Code of Conduct](./CODE_OF_CONDUCT.md)**. Please
familiarize yourself with our Code of Conduct that lists the expected behaviours.

Every contributor is expected to adhere to our Code of Conduct. It outlines our expectations and
ensures a safe, respectful environment for everyone.

----

These Contributing Guidelines have been adapted from
the [Contributing Guidelines](https://github.com/the-turing-way/the-turing-way/blob/main/CONTRIBUTING.md#recognising-contributions)
of [The Turing Way](https://github.com/the-turing-way/the-turing-way)! (License: CC-BY)
