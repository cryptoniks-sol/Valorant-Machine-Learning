const { Router } = require("express");
const router = new Router();

router.get("/", (req, res) => {
  const data = {
    contact: "n0step.xyz",
    documentation: "soon",
  };
  res.json(data);
});

module.exports = router;
