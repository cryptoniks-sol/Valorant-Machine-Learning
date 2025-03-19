const { Router } = require("express");
const router = Router();
const teamStatsController = require("../../../controllers/teamStatsController");

router.get("/:id", teamStatsController.getTeamStats);

module.exports = router;