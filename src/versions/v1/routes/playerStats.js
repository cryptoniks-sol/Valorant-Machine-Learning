const { Router } = require("express");
const router = Router();
const playerStatsController = require("../../../controllers/playerStatsController");

router.get("/:player_name", playerStatsController.getPlayerStats);

module.exports = router;
