// Save this as src/test.js in your project root
const matchDetailsService = require('./services/matchDetailsService');

// Use a match ID that's already been played and has complete data
const testMatchId = process.argv[2] || '474006';

async function testMatchDetails() {
  try {
    console.log(`Testing match details for ID: ${testMatchId}...`);
    const result = await matchDetailsService.getMatchDetails(testMatchId);
    
    console.log('\nMatch basic info:', {
      event: result.event,
      tournament: result.tournament,
      date: result.date,
      teams: result.teams.map(t => t.name)
    });
    
    console.log('\nTeam scores:', {
      [result.teams[0].name]: result.teams[0].score,
      [result.teams[1].name]: result.teams[1].score
    });
    
    console.log('\nMap results:');
    result.maps.forEach((map, index) => {
      console.log(`Map ${index+1} (${map.name}): ${map.scores.join('-')}`);
      console.log(`  Team 1 (${result.teams[0].name}) rounds - Attack: ${map.rounds.team1.attack}, Defense: ${map.rounds.team1.defense}`);
      console.log(`  Team 2 (${result.teams[1].name}) rounds - Attack: ${map.rounds.team2.attack}, Defense: ${map.rounds.team2.defense}`);
    });
    
    console.log('\nPlayer statistics:');
    result.playerStats.forEach(player => {
      console.log(`${player.name} (${player.team}): K/D/A = ${player.kills}/${player.deaths}/${player.assists}`);
    });
    
    console.log('\nTest completed successfully!');
  } catch (error) {
    console.error('Error testing match details:', error.message);
  }
}

testMatchDetails();