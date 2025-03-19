const playerStatsService = require("../services/playerStatsService");
const catchError = require("../utils/catchError");

const getPlayerStats = async (req, res) => {
  try {
    const { player_name } = req.params;
    if (!player_name) {
      return res.status(400).json({ status: "ERROR", message: "Player name is required" });
    }

    const stats = await playerStatsService.getPlayerStats(player_name);

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
