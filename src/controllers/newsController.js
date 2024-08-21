const newsService = require("../services/newsService");
const catchError = require("../utils/catchError");

const getNews = async (req, res) => {
  const page = req.query.page || 1;
  try {
    const { size, news } = await newsService.getNews(page);

    res.status(200).json({
      status: "OK",
      size,
      data: news,
    });
  } catch (error) {
    catchError(res, error);
  }
};

module.exports = {
  getNews,
};
