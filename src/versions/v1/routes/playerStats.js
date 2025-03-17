const { Router } = require("express");
const router = Router();
const playerStatsController = require("../../../controllers/playerStatsController");

router.get("/", playerStatsController.getPlayerStats);

module.exports = router;
