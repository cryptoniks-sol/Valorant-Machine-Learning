const playerStatsService = require("../services/playerStatsService");
const catchError = require("../utils/catchError");

const getPlayerStats = async (req, res) => {
  try {
    const { player } = req.query;
    if (!player) {
      return res.status(400).json({ status: "ERROR", message: "Player name is required" });
    }

    const stats = await playerStatsService.getPlayerStats(player);

    res.status(200).json({
      status: "OK",
      data: stats,
    });
  } catch (error) {
    catchError(error, res);
  }
};

module.exports = {
  getPlayerStats,
};
