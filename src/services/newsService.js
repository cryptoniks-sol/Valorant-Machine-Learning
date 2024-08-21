const request = require("request-promise");
const cheerio = require("cheerio");
const { vlrgg_url } = require("../constants");

async function getNews(page) {
  const $ = await request({
    uri: `${vlrgg_url}/news?page=${page}`,
    transform: (body) => cheerio.load(body),
  });

  const news = [];

  $(".wf-module-item").each((index, element) => {
    const article = {};
    
    article.title = $(element).find(".wf-title").text().trim();
    article.description = $(element)
      .find("div:nth-child(2)")
      .text()
      .trim();
    const dateAuthorText = $(element).find(".ge-text-light").text().trim();
    article.date = dateAuthorText.split("•")[1].trim();
    article.author = dateAuthorText.split("by")[1].trim();
    article.url_path = vlrgg_url + $(element).attr("href");

    news.push(article);
  });

  return {
    size: news.length,
    news,
  };
}

module.exports = {
  getNews,
};
