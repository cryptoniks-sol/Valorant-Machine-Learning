const { Router } = require("express");
const router = Router();
const rankingsController = require("../../../controllers/rankingsController");

router.get("/", rankingsController.getRankings);

module.exports = router;
