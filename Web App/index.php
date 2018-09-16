<html>
    <head>
        <script
			  src="https://code.jquery.com/jquery-3.3.1.min.js"
			  integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8="
			  crossorigin="anonymous"></script>
        <script type="text/javascript" src="index.js"></script>
        <title>hackathon</title>
        <link rel="stylesheet" href="style.css">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">
        <script type="text/javascript" src="script.js"></script>
    </head>
    <body>
        <div id="mySidenav" class="sidenav">
			  <a href="javascript:void(0)" class="closebtn">&times;</a>
			  <a href="#">About</a>
			  <a href="#">Services</a>
			  <a href="#">Clients</a>
			  <a href="#">Contact</a>
		</div>
		<div id="main">
            <div class="Header">
                <span class="MainHeader" onclick="openNav()">
                    &#9776; გამარჯობა, <span id="Name"></span>
                </span>
                
                <button id="CheckoutBTN" style="display: none;" onclick="Checkout()">
                    შეძენა
                </button>
            </div>
			<div id="Paragraph"></div>
		</div>
			<div class="Footer"></div>
    </body>
</html>