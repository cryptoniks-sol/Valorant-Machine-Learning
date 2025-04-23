// src/controllers/playerStatsController.js
const playerStatsService = require("../services/playerStatsService");



// Use a simple error handler function instead of importing catchError
const handleError = (error, res) => {
  console.error("Error:", error.message);
  res.status(500).json({
    status: "ERROR",
    message: error.message || "An unexpected error occurred"
  });
};

// Original function
// Updated getPlayerStats controller function
const getPlayerStats = async (req, res) => {
  const { player_name } = req.params;
  
  // Log the request for debugging
  console.log(`Getting stats for player: ${player_name}`);

  try {
    // Ensure player_name is properly decoded
    const decodedPlayerName = decodeURIComponent(player_name.trim());
    console.log(`Decoded player name: ${decodedPlayerName}`);
    
    // Try to get player stats from the stats page
    const playerStats = await playerStatsService.getPlayerStats(decodedPlayerName);
    
    if (playerStats.status === "OK" && playerStats.data) {
      return res.status(200).json(playerStats);
    }
    
    // If player not found in stats, try to get their ID via search and use detailed profile
    console.log(`Player not found in stats, trying alternative methods...`);
    const playerId = await playerStatsService.getPlayerIdByName(decodedPlayerName);
    
    if (playerId) {
      console.log(`Found player ID: ${playerId}, fetching detailed stats...`);
      const detailedStats = await playerStatsService.getDetailedPlayerStats(playerId, decodedPlayerName);
      return res.status(200).json(detailedStats);
    }
    
    // If we still can't find the player, return 404
    console.log(`Player "${decodedPlayerName}" not found by any method`);
    return res.status(404).json({
      status: "ERROR",
      message: `Player "${decodedPlayerName}" not found`
    });
  } catch (error) {
    console.error(`Error in getPlayerStats for ${player_name}:`, error);
    handleError(error, res);
  }
};

// New function - get detailed player stats by ID
const getDetailedPlayerStats = async (req, res) => {
  const { playerId } = req.params;
  const playerName = req.query.name || 'player';

  try {
    const playerStats = await playerStatsService.getDetailedPlayerStats(playerId, playerName);
    
    res.status(200).json(playerStats);
  } catch (error) {
    handleError(error, res);
  }
};

// New function - search for player by name
const searchPlayer = async (req, res) => {
  const { player_name } = req.params;

  try {
    const playerId = await playerStatsService.getPlayerIdByName(player_name);
    
    if (playerId) {
      res.status(200).json({
        status: "OK",
        data: { id: playerId, name: player_name }
      });
    } else {
      res.status(404).json({
        status: "ERROR",
        message: "Player not found"
      });
    }
  } catch (error) {
    handleError(error, res);
  }
};

// New function - get detailed player stats by name (convenience endpoint)
const getPlayerDetailsByName = async (req, res) => {
  const { player_name } = req.params;

  try {
    // First find the player ID
    const playerId = await playerStatsService.getPlayerIdByName(player_name);
    
    if (!playerId) {
      return res.status(404).json({
        status: "ERROR",
        message: "Player not found"
      });
    }
    
    // Then fetch detailed stats
    const playerStats = await playerStatsService.getDetailedPlayerStats(playerId, player_name);
    res.status(200).json(playerStats);
  } catch (error) {
    handleError(error, res);
  }
};

module.exports = {
  getPlayerStats,
  getDetailedPlayerStats,
  searchPlayer,
  getPlayerDetailsByName
};