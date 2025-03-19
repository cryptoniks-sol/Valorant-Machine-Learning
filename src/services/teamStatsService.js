const request = require("request-promise");
const cheerio = require("cheerio");
const { vlrgg_url } = require("../constants");

/**
 * Retrieves and parses team stats from the VLR website.
 * @param {string} teamId - The ID of the team.
 * @returns {Object} An object containing the team's stats.
 */
async function getTeamStats(teamId) {
  const url = `${vlrgg_url}/team/stats/${teamId}`;
  const $ = await request({ uri: url, transform: (body) => cheerio.load(body) });

  const stats = [];

  const headings = [];
  $(".wf-card.mod-dark.mod-table.mod-scroll table thead th").each((_, th) => {
    const heading = $(th).text().trim().replace(/\t+/g, "");
    headings.push(heading);
  });

  let currentMap = null;
  $(".wf-card.mod-dark.mod-table.mod-scroll table tbody tr").each((_, row) => {
    const rowData = {};

    $(row)
      .find("td")
      .each((index, td) => {
        const key = headings[index];
        let value = $(td).text().trim().replace(/\t+/g, "");

        if (key === "Agent Compositions") {
          const agents = [];
          $(td)
            .find("img")
            .each((_, img) => {
              const agentSrc = $(img).attr("src");
              const agentName = agentSrc.split("/").pop().replace(".png", "");
              agents.push(agentName);
            });
          value = agents;
        }

        if (key === "Expand") {
          value = value.replace(/\n+/g, " ").trim();
        }

        rowData[key] = value;
      });

    if (rowData["Map (#)"]) {
      currentMap = {
        map: rowData["Map (#)"],
        stats: [rowData],
      };
      stats.push(currentMap);
    } else if (currentMap) {
      currentMap.stats.push(rowData);
    }
  });

  return stats;
}

module.exports = {
  getTeamStats,
};