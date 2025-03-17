const request = require("request-promise");
const cheerio = require("cheerio");
const { vlrgg_url } = require("../constants");

/**
 * Retrieves and parses player statistics from the VLR website.
 * @param {string} playerName - The name of the player.
 * @returns {Object} An object containing the player's stats.
 */
async function getPlayerStats(playerName) {
  const url = `${vlrgg_url}/stats`;
  const $ = await request({ uri: url, transform: (body) => cheerio.load(body) });

  const stats = [];
  
  $("table tbody tr").each((index, element) => {
    const player = $(element).find("td.mod-player .text-of").text().trim();
    if (player.toLowerCase() !== playerName.toLowerCase()) return;

    const country = $(element).find(".stats-player-country").text().trim();
    const agents = $(element).find("td.mod-agents img").map((_, img) => $(img).attr("src")).get();
    const data = $(element).find("td.mod-color-sq span").map((_, span) => $(span).text().trim()).get();
    
    const [
      rating, acs, kd, kast, adr, kpr, apr, fkpr, fdpr, hs, cl, clRatio, kmax, kills, deaths, assists, fk, fd
    ] = data;

    stats.push({
      player,
      country,
      agents,
      stats: {
        rating, acs, kd, kast, adr, kpr, apr, fkpr, fdpr, hs, cl, clRatio, kmax, kills, deaths, assists, fk, fd
      }
    });
  });

  if (stats.length === 0) {
    throw new Error("Player not found");
  }

  return stats[0];
}

module.exports = {
  getPlayerStats,
};
