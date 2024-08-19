const { Router } = require("express");
const router = new Router();

router.get("/", (req, res) => {
  const data = {
    contact: "https//n0step.xyz",
    documentation: "https://statsvlr-docs.vercel.app/",
  };
  res.json(data);
});

module.exports = router;
