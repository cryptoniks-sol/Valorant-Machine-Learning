const request = require("request-promise");
const cheerio = require("cheerio");
const { vlrgg_url } = require("../constants");

/**
 * Gets detailed information about a specific match including round statistics
 * @param {string} matchId - The VLR match ID
 * @returns {Object} Detailed match information including maps, scores, and round data
 */
async function getMatchDetails(matchId) {
  if (!matchId) throw new Error("Match ID is required");

  try {
    console.log(`Fetching match details for ${matchId}...`);
    const $ = await request({
      uri: `${vlrgg_url}/${matchId}`,
      transform: (body) => cheerio.load(body),
    });

    // Detect match status
    const statusText = $(".match-header-vs-note .match-header-vs-note-span").text().trim().toLowerCase();
    const status = statusText === "final"
      ? "completed"
      : statusText === "live"
      ? "live"
      : "upcoming";

    // Get basic match info
    const matchInfo = {
      id: matchId,
      event: $(".match-header-event-series").text().trim(),
      tournament: $(".match-header-event").text().trim()
        .replace($(".match-header-event-series").text().trim(), "").trim(),
      date: $(".match-header-date").text().trim(),
      status,
      teams: [],
      maps: []
    };

    // Get team information
    let teamsFound = false;
    $(".match-header-vs-team").each((idx, el) => {
      const name = $(el).find(".match-header-vs-team-name").text().trim();
      if (name) {
        teamsFound = true;
        matchInfo.teams.push({
          name,
          score: $(el).find(".match-header-vs-team-score").text().trim(),
          logo: $(el).find(".match-header-vs-team-logo img").attr("src"),
          country: $(el).find(".flag").attr("class")
            ? $(el).find(".flag").attr("class").split(" ")[1].replace("mod-", "")
            : null
        });
      }
    });
    if (!teamsFound) {
      // upcoming formats...
      $(".wf-title-med").each((i, el) => {
        const name = $(el).text().trim();
        if (name && name !== "TBD") {
          matchInfo.teams.push({ name, score: "0", logo: null, country: null });
        }
      });
      if (!matchInfo.teams.length) {
        $(".match-header-link").each((i, el) => {
          const name = $(el).text().trim();
          if (name && name !== "TBD") {
            matchInfo.teams.push({ name, score: "0", logo: null, country: null });
          }
        });
      }
    }

    // Extract map information
    const mapContainers = $(".vm-stats-container");
    
    if (mapContainers.length > 0) {
      mapContainers.each((mapIndex, container) => {
        // Get the map name
        const mapName = $(container).find(".map-name").text().trim();
        
        const mapData = {
          name: mapName || `Map ${mapIndex + 1}`,
          scores: [],
          rounds: {
            team1: {
              attack: 0,
              defense: 0
            },
            team2: {
              attack: 0,
              defense: 0
            }
          },
          detailedRounds: []
        };
        
        // Get scores for this map
        $(container).find(".score").each((scoreIndex, scoreElement) => {
          const score = $(scoreElement).text().trim();
          mapData.scores.push(score);
        });
        
        // Get round data if available
        const roundsContainer = $(container).find(".vm-stats-game-score-round");
        if (roundsContainer.length > 0) {
          // Get the halftime score
          const halftimeElement = $(container).find(".mod-half");
          if (halftimeElement.length) {
            const halftimeText = halftimeElement.text().trim();
            const halftimeScore = halftimeText.split(":");
            const firstHalfTeam1Score = parseInt(halftimeScore[0] || "0");
            const firstHalfTeam2Score = parseInt(halftimeScore[1] || "0");
            
            // Determine which team started on which side (simplified)
            let team1StartedOnAttack = false;
            
            // Look for attack/defense icon classes to determine starting sides
            const allTeamIcons = [];
            $(container).find(".mod-t, .mod-ct").each((i, el) => {
              allTeamIcons.push($(el).attr("class"));
            });
            
            if (allTeamIcons.length > 0) {
              // The first team's side in the first half
              team1StartedOnAttack = allTeamIcons[0] && allTeamIcons[0].includes("mod-t");
            } else {
              // Fallback: assume first team on attack if it has more points at halftime
              team1StartedOnAttack = firstHalfTeam1Score > firstHalfTeam2Score;
            }
            
            // Process each round
            $(container).find(".vm-stats-game-score-round-inner .mod-win").each((roundIndex, roundElement) => {
              const roundNumber = roundIndex + 1;
              const isFirstHalf = roundNumber <= 12;
              const teamClass = $(roundElement).attr("class") || "";
              
              // Determine which team won the round
              const team1Won = teamClass.includes("mod-t");
              const team2Won = teamClass.includes("mod-ct");
              
              // Create a round object with detailed information
              const roundDetail = {
                round: roundNumber,
                winner: team1Won ? (matchInfo.teams[0] ? matchInfo.teams[0].name : "Team 1") : 
                        (matchInfo.teams[1] ? matchInfo.teams[1].name : "Team 2"),
                winnerSide: null,
                economy: {
                  team1: null,
                  team2: null
                }
              };
              
              if (team1Won) {
                // Team 1 won the round
                if ((isFirstHalf && team1StartedOnAttack) || (!isFirstHalf && !team1StartedOnAttack)) {
                  mapData.rounds.team1.attack++;
                  roundDetail.winnerSide = "attack";
                } else {
                  mapData.rounds.team1.defense++;
                  roundDetail.winnerSide = "defense";
                }
              } else if (team2Won) {
                // Team 2 won the round
                if ((isFirstHalf && team1StartedOnAttack) || (!isFirstHalf && !team1StartedOnAttack)) {
                  mapData.rounds.team2.defense++;
                  roundDetail.winnerSide = "defense";
                } else {
                  mapData.rounds.team2.attack++;
                  roundDetail.winnerSide = "attack";
                }
              }
              
              // Get win condition and economic data if available
              const roundDetailElement = $(container).find(`.vlr-rounds-row-col[data-game-id="${mapIndex}"][data-round-num="${roundNumber}"]`);
              if (roundDetailElement.length) {
                // Check for spike plant or defuse
                if (roundDetailElement.find(".vlr-rounds-row-col-score-inner.mod-bomb").length) {
                  if (roundDetail.winnerSide === "attack") {
                    roundDetail.winCondition = "spike_detonated";
                  } else {
                    roundDetail.winCondition = "spike_defused";
                  }
                } else {
                  // If no bomb indicator, it was won by eliminations
                  roundDetail.winCondition = "elimination";
                }
                
                // Get economy information
                roundDetailElement.find(".vlr-rounds-row-col-stat").each((statIndex, statElement) => {
                  const teamNumber = statIndex % 2;
                  const statType = $(statElement).find(".stat-heading").text().trim();
                  const statValue = $(statElement).find(".stat-value").text().trim();
                  
                  if (statType.includes("Loadout")) {
                    if (teamNumber === 0) {
                      roundDetail.economy.team1 = statValue;
                    } else {
                      roundDetail.economy.team2 = statValue;
                    }
                  }
                });
              }
              
              mapData.detailedRounds.push(roundDetail);
            });
          }
        }
        
        matchInfo.maps.push(mapData);
      });
    } else {
      // Try to get maps from the score overview
      $(".vm-stats-game").each((mapIndex, mapElement) => {
        const mapName = $(mapElement).find(".map-name").text().trim();
        
        const mapData = {
          name: mapName || `Map ${mapIndex + 1}`,
          scores: [],
          rounds: {
            team1: {
              attack: 0,
              defense: 0
            },
            team2: {
              attack: 0,
              defense: 0
            }
          }
        };
        
        // Get scores
        $(mapElement).find(".score").each((scoreIndex, scoreElement) => {
          const score = $(scoreElement).text().trim();
          mapData.scores.push(score);
        });
        
        matchInfo.maps.push(mapData);
      });
      
      // If still no maps, check for map picks
      if (matchInfo.maps.length === 0) {
        $(".match-header-note").each((index, element) => {
          const text = $(element).text().trim();
          if (text.includes("Map") || text.includes("map")) {
            const mapParts = text.split(/,|\//).map(part => part.trim());
            
            mapParts.forEach((part, idx) => {
              // Extract map name from formats like "Map 1: Haven" or "1. Lotus"
              const mapNameMatch = part.match(/(Map \d+:)?\s*(\d+\.\s*)?([A-Za-z]+)/);
              if (mapNameMatch && mapNameMatch[3]) {
                const mapName = mapNameMatch[3].trim();
                
                matchInfo.maps.push({
                  name: mapName,
                  scores: [],
                  rounds: {
                    team1: { attack: 0, defense: 0 },
                    team2: { attack: 0, defense: 0 }
                  }
                });
              }
            });
          }
        });
      }
    }

    // Extract player statistics
    console.log("Extracting player statistics...");
    const playerStats = [];

    // Use tableIndex to assign team correctly
    $(".wf-table-inset.mod-overview").each((tableIndex, tableElement) => {
      const teamName = matchInfo.teams[tableIndex]?.name || `Team ${tableIndex + 1}`;

      $(tableElement).find("tbody tr").each((rowIndex, rowElement) => {
        const playerName = $(rowElement).find(".mod-player .text-of").text().trim();
        if (!playerName) return;

        const kills   = parseInt($(rowElement).find(".mod-vlr-kills").text(), 10) || 0;
        const deaths  = parseInt($(rowElement).find(".mod-vlr-deaths  .side.mod-both").text(), 10) || 0;
        const assists = parseInt($(rowElement).find(".mod-vlr-assists").text(), 10) || 0;
        const acs     = parseInt($(rowElement).find(".mod-stat").eq(2).text(), 10) || 0;
        const agent   = $(rowElement).find(".mod-agents .mod-agent img").attr("alt") || "";

        playerStats.push({
          name:         playerName,
          team:         teamName,
          agent,
          acs,
          kills,
          deaths,
          assists,
          kd_ratio:     kills / Math.max(1, deaths)
        });
      });
    });

    
    // If no player stats found, try a more direct approach
    if (!playerStats.length) {
        console.log("No player stats found in tables. Trying direct extraction...");
      
      // Process all player rows
      $(".mod-player").each((playerIndex, playerElement) => {
        try {
          const teamIndex = Math.floor(playerIndex / 5);
          const playerName = $(playerElement).find(".text-of").text().trim();
          if (!playerName) return;
          
          const teamName = teamIndex < matchInfo.teams.length ? 
                          matchInfo.teams[teamIndex].name : 
                          `Team ${teamIndex + 1}`;
          
          // Get agent
          const agent = $(playerElement).find(".mod-agent").text().trim();
          
          // Get ACS
          const acs = extractFirstNumber($(playerElement).find(".mod-acs").text()) || "0";
          
          // Try to get K/D/A using various methods
          let kills = extractFirstNumber($(playerElement).find(".mod-vlr-kills").text());
          let deaths = extractFirstNumber($(playerElement).find(".mod-vlr-deaths  .side.mod-both").text());
          let assists = extractFirstNumber($(playerElement).find(".mod-vlr-assists").text());
          
          // Fallback to K/D text
          if (!kills || kills === "0") {
            const kdText = $(playerElement).find(".mod-kd").text().trim();
            if (kdText.includes("/")) {
              const parts = kdText.split("/");
              kills = extractFirstNumber(parts[0]);
              deaths = extractFirstNumber(parts[1]);
            }
            assists = extractFirstNumber($(playerElement).find(".mod-assists").text());
          }
          
          // Clean up values
          kills = sanitizeNumber(kills) || "0";
          deaths = sanitizeNumber(deaths) || "0";
          assists = sanitizeNumber(assists) || "0";
          
          console.log(`Direct - Player: ${playerName}, Team: ${teamName}, K/D/A: ${kills}/${deaths}/${assists}`);
          
          // Add player to the array
          playerStats.push({
            name: playerName,
            team: teamName,
            agent,
            acs,
            kills,
            deaths, 
            assists,
            adr: "0",
            headshot_percent: "0",
            first_kills: "0", 
            first_deaths: "0",
            maps: []
          });
        } catch (error) {
          console.error(`Error in direct player stat extraction: ${error.message}`);
        }
      });
    }
    
    // Remove duplicate players
    const uniquePlayers = [];
    const playerNames = new Set();
    
    playerStats.forEach(player => {
      if (!playerNames.has(player.name)) {
        playerNames.add(player.name);
        uniquePlayers.push(player);
      }
    });
    
    // Filter out players from duplicate teams 
    // For example, if we have Team 3, Team 4, etc. when we already have the first two teams
    const relevantPlayers = [];
    
    if (matchInfo.teams.length >= 2) {
      const teamNames = matchInfo.teams.map(team => team.name);
      uniquePlayers.forEach(player => {
        if (teamNames.includes(player.team)) {
          relevantPlayers.push(player);
        }
      });
    } else {
      relevantPlayers.push(...uniquePlayers);
    }
    
   // Modify the section where player stats are processed, after collecting them

// After collecting all playerStats, process them to keep only the total stats
console.log("Processing player statistics...");

// Step 1: Group players by name and find their total stats
const playerTotals = {};
const teamMapping = {}; // Maps Team 3 -> Titan Esports Club, Team 4 -> Wolves Esports

// First pass: create the team mapping
if (matchInfo.teams.length >= 2) {
  // Determine which numerical teams (Team 3, Team 4) correspond to which actual teams
  const mainTeams = matchInfo.teams.map(team => team.name);
  
  playerStats.forEach(player => {
    if (player.team.startsWith('Team ')) {
      const teamNum = parseInt(player.team.split(' ')[1]);
      if (teamNum === 3 || teamNum === 4) {
        // Find which main team this player belongs to based on name
        const matchingPlayers = playerStats.filter(p => 
          p.name === player.name && 
          mainTeams.includes(p.team)
        );
        
        if (matchingPlayers.length > 0) {
          teamMapping[player.team] = matchingPlayers[0].team;
        }
      }
    }
  });
}

console.log("Team mapping:", teamMapping);

// Second pass: Process players and use the total stats
playerStats.forEach(player => {
  if (player.team === 'Team 3' || player.team === 'Team 4') {
    // This is a total stat entry, keep it but with the correct team name
    const properTeamName = teamMapping[player.team] || player.team;
    
    playerTotals[player.name] = {
      name: player.name,
      team: properTeamName,
      agent: player.agent,
      acs: parseInt(player.acs) || 0,
      kills: parseInt(player.kills) || 0,
      deaths: parseInt(player.deaths) || 0,
      assists: parseInt(player.assists) || 0,
      kd_ratio: player.kd_ratio
    };
  }
});

// Replace the player stats with only the total stats
matchInfo.playerStats = Object.values(playerTotals);

// If we don't have any "total" stats, fall back to the original stats
if (matchInfo.playerStats.length === 0) {
  console.log("No total stats found, using original player stats...");
  
  // Filter to only include players from the main teams
  const mainTeams = matchInfo.teams.map(team => team.name);
  matchInfo.playerStats = playerStats.filter(player => 
    mainTeams.includes(player.team)
  );
}

console.log(`Processed ${matchInfo.playerStats.length} player statistics`);

    
    
    // Calculate additional team statistics
    matchInfo.teams.forEach((team, teamIndex) => {
      // Add rounds won on attack/defense to team objects
      team.roundsWon = {
        attack: teamIndex === 0 ? 
          matchInfo.maps.reduce((sum, map) => sum + map.rounds.team1.attack, 0) : 
          matchInfo.maps.reduce((sum, map) => sum + map.rounds.team2.attack, 0),
        defense: teamIndex === 0 ? 
          matchInfo.maps.reduce((sum, map) => sum + map.rounds.team1.defense, 0) : 
          matchInfo.maps.reduce((sum, map) => sum + map.rounds.team2.defense, 0)
      };
      
      // Calculate team's first blood success rate if the data is available
      const teamPlayers = matchInfo.playerStats.filter(player => player.team === team.name);
      const totalFirstKills = teamPlayers.reduce((sum, player) => sum + parseInt(player.first_kills || 0), 0);
      const totalFirstDeaths = teamPlayers.reduce((sum, player) => sum + parseInt(player.first_deaths || 0), 0);
      
      team.firstBloodStats = {
        kills: totalFirstKills,
        deaths: totalFirstDeaths,
        ratio: totalFirstKills / (totalFirstDeaths || 1),
        success_rate: totalFirstKills / (totalFirstKills + totalFirstDeaths || 1) * 100
      };
    });
    
    // Add additional map statistics for easier consumption
    matchInfo.maps.forEach(map => {
      // Calculate attack/defense win rates for each team
      if (map.rounds.team1.attack + map.rounds.team2.attack > 0) {
        map.attackWinRate = {
          team1: (map.rounds.team1.attack / (map.rounds.team1.attack + map.rounds.team2.defense || 1)) * 100,
          team2: (map.rounds.team2.attack / (map.rounds.team2.attack + map.rounds.team1.defense || 1)) * 100,
          overall: ((map.rounds.team1.attack + map.rounds.team2.attack) / 
            (map.rounds.team1.attack + map.rounds.team2.attack + map.rounds.team1.defense + map.rounds.team2.defense || 1)) * 100
        };
      }
      
      // Get pistol round results if available
      if (map.detailedRounds && map.detailedRounds.length >= 13) {
        map.pistolRounds = {
          firstHalf: map.detailedRounds[0].winner,
          secondHalf: map.detailedRounds[12] ? map.detailedRounds[12].winner : null
        };
      }
    });

    return matchInfo;

  } catch (error) {
    console.error("Error fetching match details:", error.message);
    throw new Error(`Failed to fetch match details: ${error.message}`);
  }
}

module.exports = { getMatchDetails };
