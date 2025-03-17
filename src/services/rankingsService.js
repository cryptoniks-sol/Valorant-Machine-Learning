const request = require("request-promise");
const cheerio = require("cheerio");
const { vlrgg_url } = require("../constants");

/**
 * Retrieves and parses team rankings from the VLR website.
 * @returns {Array} An array containing the rankings of teams.
 */
async function getRankings() {
  const url = `${vlrgg_url}/rankings`;
  const $ = await request({ uri: url, transform: (body) => cheerio.load(body) });

  let rankings = [];

  $(".world-rankings-col").each((_, region) => {
    const regionName = $(region).find("h2").text().trim();

    $(region)
      .find("table tbody tr")
      .each((_, row) => {
        const rank = $(row).find("td.rank-item-rank a").text().trim();
        const teamName = $(row).find("td.rank-item-team a div:first-child").text().trim();
        const country = $(row).find("td.rank-item-team a div.rank-item-team-country").text().trim();
        const rating = $(row).find("td.rank-item-rating a").text().trim();
        const logoUrl = $(row).find("td.rank-item-team img").attr("src");

        rankings.push({
          rank: parseInt(rank),
          team: teamName,
          country,
          rating: parseInt(rating),
          region: regionName,
          logo: logoUrl ? `https:${logoUrl}` : null
        });
      });
  });

  return rankings;
}

module.exports = {
  getRankings,
};
