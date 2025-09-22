const { Octokit } = require("@octokit/rest");

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

const owner = process.env.REPO_OWNER;
const repo = process.env.REPO_NAME.split('/').pop(); // Handles owner/repo format

const now = new Date();
const sixMonthsAgo = new Date(now);
sixMonthsAgo.setMonth(now.getMonth() - 6);
const oneMonthAgo = new Date(now);
oneMonthAgo.setMonth(now.getMonth() - 1);

async function closeOldIssues() {
  let page = 1;
  let closedCount = 0;

    // write a multiline comment into a variable:
    let body = `### Issue Cleanup: Helping Us Focus on Current Challenges

We're [reviewing](https://github.com/NVIDIA/nccl/discussions/1761) older issues to ensure we prioritize the most relevant and active ones. Since this issue hasn't seen updates in over 6 months, we'll be closing it for now.

*This change helps us focus our efforts on addressing any current issues our users are facing.* If this issue still affects you, please don't hesitate to reopen it with a quick update (e.g., \"Still relevant on [version=X]\").
Thanks for your understanding and for contributing to NCCL.`;

  while (true) {
    const { data: issues } = await octokit.issues.listForRepo({
      owner,
      repo,
      state: "open",
      per_page: 100,
      page,
    });

    if (issues.length === 0) break;

    for (const issue of issues) {
      // Ignore PRs
      if (issue.pull_request) continue;

      // Ignore issues with label "ongoing"
      if (issue.labels.some(label => label.name === "ongoing")) continue;

      const createdAt = new Date(issue.created_at);
      const updatedAt = new Date(issue.updated_at);

        if (createdAt < sixMonthsAgo && updatedAt < sixMonthsAgo) {

        // Add a comment before closing
        await octokit.issues.createComment({
          owner,
          repo,
          issue_number: issue.number,
          body: body,
        });

        await octokit.issues.update({
          owner,
          repo,
          issue_number: issue.number,
          state: "closed",
          state_reason: "not_planned",
        });
        closedCount++;
        console.log(`Closed issue #${issue.number}`);

        // Break out if we have closed 100 issues
        if (closedCount >= 100) {
          console.log("Closed 100 issues, stopping.");
          return;
        }
      }
    }
    page++;
  }
  console.log(`Total closed: ${closedCount}`);
}

closeOldIssues().catch(console.error);
