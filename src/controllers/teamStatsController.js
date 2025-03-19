const teamStatsService = require("../services/teamStatsService");
const catchError = require("../utils/catchError");

const getTeamStats = async (req, res) => {
  const { id } = req.params;

  try {
    const teamStats = await teamStatsService.getTeamStats(id);

    res.status(200).json({
      status: "OK",
      data: teamStats,
    });
  } catch (error) {
    catchError(error, res);
  }
};

module.exports = {
  getTeamStats,
};