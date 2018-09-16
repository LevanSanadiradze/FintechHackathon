var Name = 'ლევან';

var Companies = ["Wissol", "Socar"];
var Addresses = ["ვაჟა-ფშაველას გამზ. 21", "პეკინის გამზ. 9", "ჭავჭავაძის გამზ. 11", "გაგარინის ქ. 41"];
var Products = [{Name: "რეგულარი", Price: 2.72},
                {Name: "პრემიუმი", Price: 2.96},
                {Name: "რეგულარი", Price: 2.69},
                {Name: "პრემიუმი", Price: 2.99}];

var Combinations = [{
                        Company: 0,
                        Addresses: [0, 1],
                        Products: [0, 1]
                     },
                     {
                        Company: 1,
                        Addresses: [2, 3],
                        Products: [2, 3]
                     }];

var Persons = [{Name: "ლევან სანადირაძე", id: "01023356127"},
               {Name: "Honda Fit", id: "IO-123-UU"}];

var Cards = [];

var Cart = [];

var Companyid = -1;
var Addressid = -1;
var Productid = -1;

var Personid = -1;
var Cardid = -1;

var Level = 0;

var SessionID;

function showName()
{
    document.getElementById("Name").innerHTML = Name;
}

function Checkout()
{
    Level = 4;
    showMenu(Level);
}

function next(data)
{
    if(Level == 0)
    {
        Companyid = data;
        showMenu(++Level);
    }
    else if(Level == 1)
    {
        Addressid = data;
        showMenu(++Level);
    }
    else if(Level == 2)
    {
        Productid = data;
        showMenu(++Level);
    }
    else if(Level == 3)
    {
        Quantity = document.getElementById("Quantity").value;
        
        let tempProd = {Companyid: Companyid,
                        Addressid: Addressid,
                        Productid: Productid,
                        Quantity: Quantity};
        Cart.push(tempProd);
        
        document.getElementById("CheckoutBTN").style.display = '';
        
        Productid = -1;
        
        Level = 2;
        showMenu(2);
    }
    else if(Level == 4)
    {
        Personid = data;
        showMenu(++Level);
    }
    else if(Level == 5)
    {
        Cardid = data;
        showMenu(++Level);
    }
}

function drawMenu(elements, type)
{
    let Container = document.getElementById("Paragraph");
    Container.innerHTML = '';
    
    if(type == 0)
    {
        for(let i = 0; i < elements.length; i++)
        {
            var tempbtn = document.createElement('button');
            tempbtn.setAttribute('class', 'FirstHeader');
            tempbtn.setAttribute('onclick', 'next(' + i + ')');
            tempbtn.innerHTML = elements[i];
            
            Container.append(tempbtn);
        }
    }
    else if(type == 1)
    {
        var temptable = document.createElement('table');
        temptable.setAttribute('width', '100%');
        
        var temptr = document.createElement('tr');
        temptr.setAttribute('class', 'FirstHeader');
        
        var temptd1 = document.createElement('td');
        var temptd2 = document.createElement('td');
        
        var input = document.createElement('input');
        input.setAttribute('type', 'number');
        input.setAttribute('id', 'Quantity');
        input.setAttribute('placeholder', 'რაოდენობა')
        temptd1.append(input);
        
        var submit = document.createElement('button');
        submit.setAttribute('id', 'QuantityBtn');
        submit.innerHTML = 'შემდეგი';
        submit.setAttribute('onclick', 'next(-1)');
        temptd2.append(submit);
        
        temptr.append(temptd1);
        temptr.append(temptd2);
        
        temptable.append(temptr);
        
        Container.append(temptable);
    }
    else if(type == 2)
    {
        Container.innerHTML = '<div style="font-size: 50px;">' + elements[0] + '</div>';
    }
}

function showMenu(level)
{
    let elements = [];
    let type = 0;

    if(level == 0)
    {
        let variable = Combinations;
        for(let i = 0; i < variable.length; i++)
        {
            elements[i] = Companies[variable[i].Company];
        }
    }
    else if(level == 1)
    {
        let variable = Combinations[Companyid].Addresses;
        for(let i = 0; i < variable.length; i++)
        {
            elements[i] = Addresses[variable[i]];
        }
    }
    else if(level == 2)
    {
        let variable = Combinations[Companyid].Products;
        for(let i = 0; i < variable.length; i++)
        {
            elements[i] = Products[variable[i]].Name + ' - ' + Products[variable[i]].Price + ' ლ';
        }
    }
    else if(level == 3)
    {
        type = 1;
    }
    else if(level == 4)
    {
        for(let i = 0; i < Persons.length; i++)
        {
            elements[i] = Persons[i].Name + ' - ' + Persons[i].id;
        }
    }
    else if(level == 5)
    {
        for(let i = 0; i < Cards.length; i++)
        {
            elements[i] = Cards[i];
        }
    }
    else if(level == 6)
    {
        doPayment();
    }
    else if(level == 7)
    {
        elements.push('გმადლობთ, გადახდა წარმატებით განხორციელდა.');
        type = 2;
    }

    drawMenu(elements, type);
}

$(document).ready(function () {
    
    apilogin();
});

function apilogin() {
    
    $.get('https://cors-anywhere.herokuapp.com/https://api.fintech.ge/api/Clients/Login/test/1234', function(response) {
        SessionID = response.SessionId;
        
        Name = response.UserDetails.Name;
        
        getCards();
        
        showName();
        showMenu(0);
    });
}

function getCards() {
    
    $.get('https://cors-anywhere.herokuapp.com/https://api.fintech.ge/api/Products/Cards/1', function(response) {
        for(let i = 0; i < response.length; i++)
            Cards.push('**** **** **** ' + response[i].LastFour);
    });
}

function doPayment(Cardid) {
    
    $.get('payment.php', function(response) {
        Level = 7;
        showMenu(Level);
    });
}