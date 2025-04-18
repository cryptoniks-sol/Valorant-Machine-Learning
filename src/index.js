const express = require("express");
const morgan = require("morgan");
let cors = require("cors");

const app = express();

// Settings
app.set("port", process.env.PORT || 5000);

// Middlewares
app.use(cors());
app.use(morgan("dev"));
app.use(express.urlencoded({ extended: false }));
app.use(express.json());

// Routes
app.use(require("./versions/v1/routes/index"));
app.use("/api", require("./versions/v1/routes/index"));
// - Version 1
app.use("/api/v1/teams", require("./versions/v1/routes/teams"));
app.use("/api/v1/players", require("./versions/v1/routes/players"));
app.use("/api/v1/events", require("./versions/v1/routes/events"));
app.use("/api/v1/matches", require("./versions/v1/routes/matches"));
app.use("/api/v1/results", require("./versions/v1/routes/results"));
app.use("/api/v1/news", require("./versions/v1/routes/news"))
app.use("/api/v1/match-history", require("./versions/v1/routes/matchHistory"));
app.use("/api/v1/player-stats", require("./versions/v1/routes/playerStats"));
app.use("/api/v1/rankings", require("./versions/v1/routes/rankings"));
app.use("/api/v1/team-stats", require("./versions/v1/routes/teamStats"));
app.use("/api/v1/match-details", require("./versions/v1/routes/matchDetailsRoutes"));
// Starting server
app.listen(app.get("port"), () => {
  console.log(`Server running on port ${app.get("port")}`);
});
