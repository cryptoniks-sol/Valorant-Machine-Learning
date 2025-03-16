const matchHistoryService = require("../services/matchHistoryService");

const getMatchHistory = async (req, res) => {
  try {
    const team = req.query.team;
    if (!team) {
      return res.status(400).json({ status: "ERROR", message: "Team name is required" });
    }

    const { size, matches } = await matchHistoryService.getMatchHistory(team);
    res.status(200).json({
      status: "OK",
      size,
      data: matches,
    });
  } catch (error) {
    res.status(500).json({ status: "ERROR", message: error.message });
  }
};

module.exports = { getMatchHistory };
