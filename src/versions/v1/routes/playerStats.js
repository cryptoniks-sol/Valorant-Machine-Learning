// src/versions/v1/routes/playerStats.js
const { Router } = require("express");
const router = Router();
const playerStatsController = require("../../../controllers/playerStatsController");

// Add a test route to check basic functionality
router.get("/test", (req, res) => {
  res.json({ status: "OK", message: "Player stats API is working" });
});

// Update original endpoint to be more flexible
router.get("/:player_name", playerStatsController.getPlayerStats);

// Add new endpoints with different URL patterns
router.get("/id/:playerId", playerStatsController.getDetailedPlayerStats);
router.get("/search/:player_name", playerStatsController.searchPlayer);
router.get("/detailed/:player_name", playerStatsController.getPlayerDetailsByName);

module.exports = router;