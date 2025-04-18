const matchDetailsService = require("../services/matchDetailsService");

/**
 * Controller to handle match details requests
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
const getMatchDetails = async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) {
      return res.status(400).json({ status: "ERROR", message: "Match ID is required" });
    }

    const matchDetails = await matchDetailsService.getMatchDetails(id);
    res.status(200).json({
      status: "OK",
      data: matchDetails,
    });
  } catch (error) {
    console.error(`Error in getMatchDetails controller: ${error.message}`);
    res.status(500).json({ status: "ERROR", message: error.message });
  }
};

module.exports = {
  getMatchDetails,
};