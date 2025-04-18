const express = require("express");
const router = express.Router();
const { getMatchDetails } = require("../../../controllers/matchDetailsController");

/**
 * @route GET /:id
 * @description Get detailed information about a specific match
 * @param {string} id - The VLR.gg match ID
 * @returns {Object} Detailed match information including maps, scores, and player stats
 */
router.get("/:id", getMatchDetails);

module.exports = router;