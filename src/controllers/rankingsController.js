const rankingsService = require("../services/rankingsService");
const catchError = require("../utils/catchError");

const getRankings = async (req, res) => {
  try {
    const region = req.query.region;
    const rankings = await rankingsService.getRankings(region);

    res.status(200).json({
      status: "OK",
      data: rankings,
    });
  } catch (error) {
    catchError(error, res);
  }
};

module.exports = {
  getRankings,
};