const teamStatsService = require("../services/teamStatsService");
const catchError = require("../utils/catchError");

const getTeamStats = async (req, res, next) => {
  try {
    const { id } = req.params;
    
    if (!id) {
      return res.status(400).json({
        status: "Error",
        message: "Team ID is required"
      });
    }
    
    const teamStats = await teamStatsService.getTeamStats(id);
    
    return res.status(200).json({
      status: "OK",
      data: teamStats,
    });
  } catch (error) {
    // Pass next to catchError or handle directly
    return next(error);
    // Or if catchError expects res:
    // return catchError(error, res, next);
  }
};

module.exports = {
  getTeamStats,
};