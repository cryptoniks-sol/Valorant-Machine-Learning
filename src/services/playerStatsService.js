// src/services/playerStatsService.js
const request = require("request-promise");
const cheerio = require("cheerio");

const vlrgg_url = "https://www.vlr.gg"; // Add this directly since we don't have a constants file

/**
 * Retrieves and parses detailed player statistics from a VLR.gg player profile page.
 * @param {string} playerId - The VLR.gg player ID.
 * @param {string} playerName - The player's name (for URL construction).
 * @returns {Object} An object containing comprehensive player stats and information.
 */
async function getDetailedPlayerStats(playerId, playerName) {
  try {
    // Construct the player profile URL
    const url = `${vlrgg_url}/player/${playerId}/${playerName}`;
    console.log(`Fetching player data from: ${url}`);
    
    const html = await request({ uri: url, transform: (body) => cheerio.load(body) });
    const $ = html;
    
    // Extract player basic info
    const playerInfo = {
      id: playerId,
      name: $(".wf-title").text().trim(),
      realName: $(".player-real-name").text().trim(),
      country: $(".ge-text-light i.flag").parent().text().trim(),
      socials: {}
    };
    
    // Extract team information
    const currentTeam = {
      name: $(".wf-module-item:first-child div:first-child").text().trim(),
      joined: $(".wf-module-item:first-child .ge-text-light").first().text().trim()
    };
    
    playerInfo.team = currentTeam;
    
    // Extract social media links
    $(".player-summary-container-1 a[href^='https://']").each((_, element) => {
      const link = $(element).attr("href");
      const text = $(element).text().trim();
      if (link.includes("twitter.com") || link.includes("x.com")) {
        playerInfo.socials.twitter = text;
        playerInfo.socials.twitter_url = link;
      } else if (link.includes("twitch.tv") || link.includes("weibo.com")) {
        playerInfo.socials.twitch = text;
        playerInfo.socials.twitch_url = link;
      }
    });
    
    // Extract agent statistics - more robust column detection
    const agentStats = [];
    
    // Find the agent stats table headers to determine column indices
    const headerIndices = {};
    $(".wf-card.mod-table.mod-dark thead th").each((index, element) => {
      const headerText = $(element).attr('title') || $(element).text().trim();
      if (headerText.includes("Agent")) headerIndices.agent = index + 1;
      if (headerText.includes("Usage")) headerIndices.usage = index + 1;
      if (headerText.includes("Rounds")) headerIndices.rounds = index + 1;
      if (headerText.includes("Rating")) headerIndices.rating = index + 1;
      if (headerText.includes("Combat Score")) headerIndices.acs = index + 1;
      if (headerText.includes("Kills:Death")) headerIndices.kd = index + 1;
      if (headerText.includes("Damage")) headerIndices.adr = index + 1;
      if (headerText.includes("KAST")) headerIndices.kast = index + 1;
      if (headerText.includes("Kills Per Round")) headerIndices.kpr = index + 1;
      if (headerText.includes("Assists Per Round")) headerIndices.apr = index + 1;
      if (headerText.includes("First Kills")) headerIndices.fkpr = index + 1;
      if (headerText.includes("First Deaths")) headerIndices.fdpr = index + 1;
      if (headerText === "K") headerIndices.kills = index + 1;
      if (headerText === "D") headerIndices.deaths = index + 1;
      if (headerText === "A") headerIndices.assists = index + 1;
      if (headerText === "FK" || headerText.includes("First Bloods")) headerIndices.fk = index + 1;
      if (headerText === "FD") headerIndices.fd = index + 1;
    });
    
    // Process each agent row
    $(".wf-card.mod-table.mod-dark tbody tr").each((_, row) => {
      const $row = $(row);
      const agent = $row.find(`td:nth-child(${headerIndices.agent || 1}) img`).attr("alt");
      
      if (!agent) return; // Skip if no agent found
      
      const stats = {
        agentName: agent,
        matchesPlayed: $row.find(`td:nth-child(${headerIndices.usage || 2})`).text().trim(),
        roundsPlayed: $row.find(`td:nth-child(${headerIndices.rounds || 3})`).text().trim(),
        rating: $row.find(`td:nth-child(${headerIndices.rating || 4})`).text().trim(),
        averageCombatScore: $row.find(`td:nth-child(${headerIndices.acs || 5})`).text().trim(),
        kd: $row.find(`td:nth-child(${headerIndices.kd || 6})`).text().trim(),
        averageDamagePerRound: $row.find(`td:nth-child(${headerIndices.adr || 7})`).text().trim(),
        kast: $row.find(`td:nth-child(${headerIndices.kast || 8})`).text().trim(),
        killsPerRound: $row.find(`td:nth-child(${headerIndices.kpr || 9})`).text().trim(),
        assistsPerRound: $row.find(`td:nth-child(${headerIndices.apr || 10})`).text().trim(),
        firstKillsPerRound: $row.find(`td:nth-child(${headerIndices.fkpr || 11})`).text().trim(),
        firstDeathsPerRound: $row.find(`td:nth-child(${headerIndices.fdpr || 12})`).text().trim(),
        kills: $row.find(`td:nth-child(${headerIndices.kills || 13})`).text().trim(),
        deaths: $row.find(`td:nth-child(${headerIndices.deaths || 14})`).text().trim(),
        assists: $row.find(`td:nth-child(${headerIndices.assists || 15})`).text().trim(),
        firstKills: $row.find(`td:nth-child(${headerIndices.fk || 16})`).text().trim(),
        firstDeaths: $row.find(`td:nth-child(${headerIndices.fd || 17})`).text().trim()
      };
      
      agentStats.push(stats);
    });
    
    // Extract recent match results
    const recentMatches = [];
    $(".wf-card.fc-flex.m-item").each((_, matchElement) => {
      const $match = $(matchElement);
      
      // Get event info
      const eventName = $match.find(".m-item-event .text-of").text().trim();
      const eventStage = $match.find(".m-item-event").text().trim().replace(eventName, '').trim();
      
      // Get date and time
      const date = $match.find(".m-item-date div").first().text().trim();
      const time = $match.find(".m-item-date").contents().last().text().trim();
      
      // Get team names and scores
      const team1Name = $match.find(".m-item-team:not(.mod-right) .m-item-team-name").text().trim();
      const team2Name = $match.find(".m-item-team.mod-right .m-item-team-name").text().trim();
      
      // Extract score from the result div
      const resultText = $match.find(".m-item-result").text().trim();
      const scores = resultText.match(/\d+/g);
      let team1Score = 0;
      let team2Score = 0;
      
      if (scores && scores.length >= 2) {
        team1Score = parseInt(scores[0], 10);
        team2Score = parseInt(scores[1], 10);
      }
      
      // Determine match result (win/loss)
      const isWin = $match.find(".m-item-result").hasClass("mod-win");
      
      // Check if the player's team is team1 or team2
      // This could be further improved by checking team logos or names against currentTeam
      const isPlayerTeam1 = true; // Default assumption
      
      recentMatches.push({
        event: eventName,
        stage: eventStage,
        date: date,
        time: time,
        team1: {
          name: team1Name,
          score: team1Score,
          isPlayerTeam: isPlayerTeam1
        },
        team2: {
          name: team2Name,
          score: team2Score,
          isPlayerTeam: !isPlayerTeam1
        },
        result: isWin ? "win" : "loss",
        matchUrl: $match.attr("href")
      });
    });
    
    // Calculate aggregate stats across all agents
    const totalMatches = agentStats.reduce((sum, agent) => {
      // Extract numeric value from format like "(13) 42%"
      const matchesMatch = agent.matchesPlayed.match(/\((\d+)\)/);
      const matches = matchesMatch && matchesMatch[1] ? parseInt(matchesMatch[1], 10) : 0;
      return sum + matches;
    }, 0);
    
    // Calculate weighted average stats
    let weightedRating = 0;
    let weightedACS = 0;
    let weightedADR = 0;
    let weightedKD = 0;
    let weightedKAST = 0;
    let weightedKPR = 0;
    let weightedAPR = 0;
    let weightedFKPR = 0;
    let weightedFDPR = 0;
    let totalWeights = 0;
    
    agentStats.forEach(agent => {
      // Extract agent usage percentage and number of matches
      const usageMatch = agent.matchesPlayed.match(/\((\d+)\)(?:\s*(\d+)%)?/);
      
      if (usageMatch) {
        const matches = parseInt(usageMatch[1], 10) || 0;
        const weight = matches / Math.max(1, totalMatches);
        totalWeights += weight;
        
        // Add weighted stats
        weightedRating += parseFloat(agent.rating || 0) * weight;
        weightedACS += parseFloat(agent.averageCombatScore || 0) * weight;
        weightedADR += parseFloat(agent.averageDamagePerRound || 0) * weight;
        weightedKD += parseFloat(agent.kd || 0) * weight;
        weightedKAST += (parseFloat(agent.kast.replace('%', '') || 0) / 100) * weight;
        weightedKPR += parseFloat(agent.killsPerRound || 0) * weight;
        weightedAPR += parseFloat(agent.assistsPerRound || 0) * weight;
        weightedFKPR += parseFloat(agent.firstKillsPerRound || 0) * weight;
        weightedFDPR += parseFloat(agent.firstDeathsPerRound || 0) * weight;
      }
    });
    
    // Adjust for total weights in case some agents didn't have valid data
    const normalizeFactor = totalWeights > 0 ? 1 / totalWeights : 1;
    
    // Compile overall performance metrics
    const overallStats = {
      totalMatches: totalMatches,
      rating: (weightedRating * normalizeFactor).toFixed(2),
      averageCombatScore: (weightedACS * normalizeFactor).toFixed(1),
      averageDamagePerRound: (weightedADR * normalizeFactor).toFixed(1),
      killDeathRatio: (weightedKD * normalizeFactor).toFixed(2),
      kastPercentage: ((weightedKAST * normalizeFactor) * 100).toFixed(1) + '%',
      killsPerRound: (weightedKPR * normalizeFactor).toFixed(2),
      assistsPerRound: (weightedAPR * normalizeFactor).toFixed(2),
      firstKillsPerRound: (weightedFKPR * normalizeFactor).toFixed(2),
      firstDeathsPerRound: (weightedFDPR * normalizeFactor).toFixed(2),
      // Win rate calculation from recent matches
      recentWinRate: recentMatches.length > 0 ? 
        (recentMatches.filter(m => m.result === "win").length / recentMatches.length).toFixed(2) : 
        "0.00"
    };
    
    // Extract past teams
    const pastTeams = [];
    $(".wf-label:contains('Past Teams')").next(".wf-card").find(".wf-module-item").each((_, element) => {
      const $team = $(element);
      const name = $team.find("div:first-child").text().trim();
      const duration = $team.find(".ge-text-light").first().text().trim();
      
      if (name && name !== currentTeam.name) {
        pastTeams.push({
          name: name,
          duration: duration
        });
      }
    });

    // Extract event placements
    const eventPlacements = [];
    $(".player-event-item").each((_, element) => {
      const $event = $(element);
      const eventName = $event.find(".text-of").text().trim();
      const year = $event.find("div:last-child").text().trim();
      
      // Get all placements for this event
      const placements = [];
      $event.find(".ge-text-light").each((_, placementElement) => {
        const placementText = $(placementElement).text().trim();
        if (placementText) {
          placements.push(placementText);
        }
      });
      
      // Check if there's prize money
      let prizeMoney = null;
      const prizeMoneyText = $event.find("span[style*='font-weight: 700']").text().trim();
      if (prizeMoneyText.length > 0) {
        prizeMoney = prizeMoneyText;
      }
      
      if (eventName) {
        eventPlacements.push({
          eventName: eventName,
          year: year,
          placements: placements,
          prizeMoney: prizeMoney
        });
      }
    });
    
    // Build result structure
    return {
      status: "OK",
      data: {
        info: playerInfo,
        agents: agentStats,
        results: recentMatches.slice(0, 5), // Limit to 5 most recent matches
        overallStats: overallStats,
        pastTeams: pastTeams,
        events: eventPlacements.slice(0, 10) // Limit to 10 most recent events
      }
    };
  } catch (error) {
    console.error(`Error fetching player stats for ${playerId}/${playerName}:`, error);
    return {
      status: "ERROR",
      message: `Failed to fetch player stats: ${error.message}`,
      data: null
    };
  }
}

/**
 * Gets the player ID from their name using the search feature
 * @param {string} playerName - The name of the player to search for
 * @returns {Promise<string|null>} The player ID if found, null otherwise
 */
async function getPlayerIdByName(playerName) {
  try {
    // First try direct search
    const url = `${vlrgg_url}/search/?q=${encodeURIComponent(playerName)}`;
    console.log(`Searching for player: ${url}`);
    
    const $ = await request({ uri: url, transform: (body) => cheerio.load(body) });
    
    // Find player in search results
    const playerLink = $("a.search-item").filter((_, element) => {
      const type = $(element).find(".search-item-type").text().trim();
      const name = $(element).find(".search-item-name").text().trim();
      return type === "Player" && name.toLowerCase().includes(playerName.toLowerCase());
    }).first();
    
    if (playerLink.length > 0) {
      // Extract player ID from URL
      const href = playerLink.attr("href");
      const match = href.match(/\/player\/(\d+)\//);
      if (match && match[1]) {
        return match[1];
      }
    }
    
    // If direct search fails, try checking on stats page
    const statsUrl = `${vlrgg_url}/stats`;
    console.log(`Checking stats page: ${statsUrl}`);
    
    const statsPage = await request({ uri: statsUrl, transform: (body) => cheerio.load(body) });
    
    let playerId = null;
    
    statsPage("table tbody tr").each((_, row) => {
      const player = statsPage(row).find("td.mod-player .text-of").text().trim();
      if (player.toLowerCase() === playerName.toLowerCase()) {
        const playerLink = statsPage(row).find("td.mod-player a").attr("href");
        if (playerLink) {
          const match = playerLink.match(/\/player\/(\d+)\//);
          if (match && match[1]) {
            playerId = match[1];
            return false; // Break the loop
          }
        }
      }
    });
    
    return playerId;
  } catch (error) {
    console.error(`Error searching for player ID: ${error.message}`);
    return null;
  }
}

/**
 * Legacy function for basic stats from VLR stats page
 */
/**
 * Legacy function for basic stats from VLR stats page with improved filtering
 */
async function getPlayerStats(playerName) {
  try {
    // Use the filtered URL to ensure we can see all players
    const url = `${vlrgg_url}/stats/?event_group_id=all&region=all&min_rounds=0&min_rating=0&agent=all&map_id=all&timespan=60d`;
    console.log(`Fetching stats from: ${url}`);
    
    const $ = await request({ uri: url, transform: (body) => cheerio.load(body) });

    let playerStats = null;
    let playerFound = false;

    $("table tbody tr").each((index, element) => {
      const player = $(element).find("td.mod-player .text-of").text().trim();
      
      // Case-insensitive comparison
      if (player.toLowerCase() !== playerName.toLowerCase()) return;
      
      playerFound = true;
      console.log(`Found player: ${player}`);

      const country = $(element).find(".stats-player-country").text().trim();
      
      const agents = $(element)
        .find("td.mod-agents img")
        .map((_, img) => $(img).attr("src").split("/").pop().replace(".png", ""))
        .get();

      const data = $(element).find("td").map((_, td) => $(td).text().trim()).get();

      // Create a more flexible data extraction approach
      const stats = {};
      let offset = 2; // Skip player and agents columns
      
      // Map column indices to stat names
      const statNames = [
        'roundsPlayed', 'rating', 'acs', 'kd', 'kast', 'adr', 'kpr', 'apr', 
        'fkpr', 'fdpr', 'hs', 'cl', 'clRatio', 'kmax', 'kills', 'deaths', 
        'assists', 'fk', 'fd'
      ];
      
      // Extract all available stats
      for (let i = 0; i < statNames.length; i++) {
        if (offset + i < data.length) {
          stats[statNames[i]] = data[offset + i];
        }
      }
      
      playerStats = {
        player,
        country,
        agents,
        stats
      };
    });

    if (!playerStats) {
      // If player not found, try to get their ID by searching
      console.log(`Player not found in stats table, trying to search...`);
      const playerId = await getPlayerIdByName(playerName);
      
      if (playerId) {
        console.log(`Found player ID: ${playerId} through search`);
        // Get detailed player data instead
        const detailedStats = await getDetailedPlayerStats(playerId, playerName);
        return detailedStats;
      }
      
      return { 
        status: "ERROR", 
        message: `Player "${playerName}" not found in stats table or search`, 
        data: null 
      };
    }

    // Try to find player ID
    const playerLink = $(`table tbody tr`).filter(function() {
      return $(this).find("td.mod-player .text-of").text().trim().toLowerCase() === playerName.toLowerCase();
    }).find("td.mod-player a").attr("href");
    
    if (playerLink) {
      const match = playerLink.match(/\/player\/(\d+)\//);
      if (match && match[1]) {
        playerStats.id = match[1];
        console.log(`Found player ID: ${playerStats.id}`);
      }
    }

    return { status: "OK", data: playerStats };
  } catch (error) {
    console.error(`Error in getPlayerStats: ${error.message}`);
    return { status: "ERROR", message: error.message, data: null };
  }
}

module.exports = {
  getPlayerStats,
  getDetailedPlayerStats,
  getPlayerIdByName
};