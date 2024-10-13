function updateClock() {
  const now = new Date();
  const hour = now.getHours() % 12; // Convert to 12-hour format
  const minute = now.getMinutes();
  const second = now.getSeconds();
  const millisecond = now.getMilliseconds();

  const hourHand = document.getElementById('hourHand');
  const secondHand = document.getElementById('secondHand');

  // Calculate rotation angles for hour and second hands
  const hourAngle = (360 / 12) * (hour + minute / 60);
  const secondAngle = ((360 / 60) * (second + millisecond / 1000)); // Including milliseconds for smooth continuous rotation

  // Apply rotation to hands
  hourHand.style.transform = `rotate(${hourAngle}deg)`;
  secondHand.style.transform = `rotate(${secondAngle}deg)`;
}

setInterval(updateClock, 16); 
updateClock();