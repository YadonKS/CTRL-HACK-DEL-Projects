const { exec } = require("child_process");

console.log("In Node.js, calling final Python script...");
exec("python end.py \"Hello, World!\"", (error, stdout, stderr) => {
  console.log(`stdout: ${stdout}`);
});