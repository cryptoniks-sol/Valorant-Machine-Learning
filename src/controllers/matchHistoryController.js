const matchHistoryService = require("../services/matchHistoryService");

const getMatchHistory = async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) {
      return res.status(400).json({ status: "ERROR", message: "Team ID is required" });
    }

    const { size, matches } = await matchHistoryService.getMatchHistory(id);
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
