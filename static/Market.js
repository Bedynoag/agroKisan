//image loop h y bhai   


var imageUrls = ["https://static.vecteezy.com/system/resources/previews/021/351/115/original/bowl-with-honey-isolated-on-a-transparent-background-png.png",
"https://cdn.pixabay.com/photo/2016/10/14/13/18/dal-1740205_1280.png","https://static.vecteezy.com/system/resources/previews/008/533/966/original/coffee-beans-in-the-bowl-cutout-file-png.png","https://pngimg.com/d/rice_PNG34.png","https://static.vecteezy.com/system/resources/previews/009/339/311/original/soy-beans-in-bowl-wood-free-png.png"
];

const myImage = document.getElementById('myImage');

function changeImageLoop() {
  let index = 0;

  function setImage() {
    myImage.src = imageUrls[index];
    index = (index + 1) % imageUrls.length; 
  }

  setImage(); 
  setInterval(setImage, 2000);
}

changeImageLoop();



// event jo ho rha h button dabane prr brr brr go krne prr

document.addEventListener("DOMContentLoaded", function() {
  const firstDiv = document.getElementById("main");
  const secondDiv = document.getElementById("load");
  const thirdDiv = document.getElementById("result");
  const fourthDiv = document.getElementById("myImage");
  const showButton = document.getElementById("btn_predict");

  
  function handleButtonClick() {
    firstDiv.style.display = "none";
    secondDiv.style.display = "block";
    thirdDiv.style.display = "none";
    fourthDiv.style.display = "block";

    setTimeout(function () {
      secondDiv.style.display = "none";
      thirdDiv.style.display = "block";
    }, 1000);

    setTimeout(function () {
      thirdDiv.style.display = "none";
      firstDiv.style.display = "block";
    }, 1000);

    
    setTimeout(function () {
      thirdDiv.style.display = "block";
      fourthDiv.style.display = "none";
    }, 1000);

   
  }

  showButton.addEventListener("click", handleButtonClick);
});


function makeFieldsRequired() {
  var form = document.querySelector('form');
  var inputElements = form.querySelectorAll('input');
  var selectElements = form.querySelectorAll('select');

  inputElements.forEach(function (input) {
      if (!input.hasAttribute('required')) {
          input.setAttribute('required', true);
      }
  });

  selectElements.forEach(function (select) {
      if (!select.hasAttribute('required')) {
          select.setAttribute('required', true);
      }
  });
}

// Call the function to make fields required
makeFieldsRequired();
