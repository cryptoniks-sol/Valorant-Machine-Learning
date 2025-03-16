const request = require("request-promise");
const cheerio = require("cheerio");
const { vlrgg_url } = require("../constants");

async function getMatchHistory(teamName) {
  if (!teamName) throw new Error("Team name is required.");

  const formattedTeamName = teamName.toLowerCase().replace(/\s+/g, "-");
  const url = `${vlrgg_url}/team/matches/13790/${formattedTeamName}/?group=completed`;

  // clean: user agent will be removed once data is debugged
  try {
    const html = await request({
      uri: url,
      transform: (body) => cheerio.load(body),
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
      }
    });

    const $ = html;
    const matches = [];

    $("div.mod-dark > div").each((index, element) => {
      const matchElement = $(element).find("a").first();
      if (!matchElement.length) return;

      const matchLink = matchElement.attr("href");
      const matchId = matchLink.split("/")[1];

      const event = matchElement.find(".m-item-event div").first().text().trim();

      const matchDate = matchElement.find(".m-item-date div").text().trim();
      const matchTime = matchElement.find(".m-item-date").text().replace(matchDate, "").trim();

      const team1 = matchElement.find(".m-item-team").first().find(".m-item-team-name").text().trim();
      const team2 = matchElement.find(".m-item-team.mod-right").find(".m-item-team-name").text().trim();

      const scoreElements = matchElement.find(".m-item-result span");
      const scoreTeam1 = scoreElements.first().text().trim();
      const scoreTeam2 = scoreElements.last().text().trim();

      matches.push({
        id: matchId,
        event,
        date: matchDate,
        time: matchTime,
        teams: [
          { name: team1, score: scoreTeam1 },
          { name: team2, score: scoreTeam2 }
        ],
        link: `${vlrgg_url}${matchLink}`
      });
    });

    return {
      size: matches.length,
      matches,
    };
  } catch (error) {
    console.error("Error fetching match history:", error.message);
    return { size: 0, matches: [] };
  }
}

module.exports = { getMatchHistory };
