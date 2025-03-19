const request = require("request-promise");
const cheerio = require("cheerio");
const { vlrgg_url } = require("../constants");

/**
 * Retrieves and parses team rankings from the VLR website.
 * @param {string} region - The region to filter rankings by (e.g., 'asia-pacific').
 * @returns {Array} An array containing the rankings of teams.
 */
async function getRankings(region) {
  let url = `${vlrgg_url}/rankings`;

  // If a region is specified and not "all", append it to the URL
  if (region && region !== "all") {
    switch (region) {
      case "na":
        url = `${vlrgg_url}/rankings/north-america`;
        break;
      case "eu":
        url = `${vlrgg_url}/rankings/europe`;
        break;
      case "br":
        url = `${vlrgg_url}/rankings/brazil`;
        break;
      case "ap":
        url = `${vlrgg_url}/rankings/asia-pacific`;
        break;
      case "asia":
        url = `${vlrgg_url}/rankings/asia-pacific`;
        break;
      case "pacific":
        url = `${vlrgg_url}/rankings/asia-pacific`;
        break;
      case "kr":
        url = `${vlrgg_url}/rankings/korea`;
        break;
      case "ch":
        url = `${vlrgg_url}/rankings/china`;
        break;
      case "jp":
        url = `${vlrgg_url}/rankings/japan`;
        break;
      case "las":
      case "lan":
        url = `${vlrgg_url}/rankings/latin-america`;
        break;
      case "oce":
        url = `${vlrgg_url}/rankings/oceania`;
        break;
      case "mena":
        url = `${vlrgg_url}/rankings/mena`;
        break;
      case "gc":
        url = `${vlrgg_url}/rankings/game-changers`;
        break;
      default:
        url = `${vlrgg_url}/rankings/${region}`;
    }
  }

  const $ = await request({ uri: url, transform: (body) => cheerio.load(body) });

  let rankings = [];

  const isRegionPage = $(".mod-scroll").length > 0;

  if (isRegionPage) {
    $(".mod-scroll .rank-item").each((_, row) => {
      const rank = $(row).find(".rank-item-rank-num").text().trim();
      const teamName = $(row).find(".rank-item-team img").attr("alt");
      const country = $(row).find(".rank-item-team-country").text().trim();
      const rating = $(row).find(".rank-item-rating").first().text().trim();
      const logoUrl = $(row).find(".rank-item-team img").attr("src");

      rankings.push({
        rank: parseInt(rank),
        team: teamName,
        country,
        rating: parseInt(rating),
        region: region || "Global",
        logo: logoUrl ? `https:${logoUrl}` : null,
      });
    });
  } else {
    $(".world-rankings-col").each((_, region) => {
      const regionName = $(region).find("h2").text().trim();

      $(region)
        .find("table tbody tr")
        .each((_, row) => {
          const rank = $(row).find("td.rank-item-rank a").text().trim();
          const teamName = $(row).find("td.rank-item-team img").attr("alt");
          const country = $(row).find("td.rank-item-team a div.rank-item-team-country").text().trim();
          const rating = $(row).find("td.rank-item-rating a").text().trim();
          const logoUrl = $(row).find("td.rank-item-team img").attr("src");

          rankings.push({
            rank: parseInt(rank),
            team: teamName,
            country,
            rating: parseInt(rating),
            region: regionName,
            logo: logoUrl ? `https:${logoUrl}` : null,
          });
        });
    });
  }

  return rankings;
}

module.exports = {
  getRankings,
};