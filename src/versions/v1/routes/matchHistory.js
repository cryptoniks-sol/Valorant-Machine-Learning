const express = require("express");
const router = express.Router();
const matchHistoryController = require("../../../controllers/matchHistoryController");

router.get("/:id", matchHistoryController.getMatchHistory);

module.exports = router;
