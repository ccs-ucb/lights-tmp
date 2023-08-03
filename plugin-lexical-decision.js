var jsPsychLightGrid = (function (jspsych) {
  "use strict";

  const info = {
    name: "LIGHT-GRID",
    parameters: {
      nrows: {
        type: jspsych.ParameterType.INT,
        default: undefined,
      },
      ncolumns: {
        type: jspsych.ParameterType.INT,
        default: undefined,
      },
    },
  };

  const addCSS = css => {
    document.head.appendChild(
      document.createElement("style")
    ).innerHTML=css;
  }

  const gridSum = grid => {
    return grid.reduce(
      function(a,b) { return a.concat(b) }
    ).reduce(function(a,b) { return a + b });
  }

  class jsPsychLightGridPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {

      // setup the game
      let gameState = {
        gridState: [] ,//binary 2d array
        
        opacityOn: .95,
        opacityOff: .25,
        ncols: trial.ncolumns,
        nrows: trial.nrows,
        actionDelta: false // turning off a light reverses it's *action*
      }

      let trialData = {}


      function renderState() {
        gameState.gridState.forEach(function (row, rowIndex) {
          row.forEach(function (col, colIndex) {
            let key = `light_${rowIndex}_${colIndex}`
            let light = document.getElementById(key)
            if (col){
              // light.style.backgroundColor = getcolor(
              let color = getcolor(
              rowIndex, colIndex, gameState.ncols, 
              gameState.nrows, gameState.opacityOn
              );
              
              gsap.to(light, { 
                  backgroundColor: color,
                  ease: "power4",
                  boxShadow: `0 0 10px ${color}`
                });
              } else {
                let color = getcolor(
                rowIndex, colIndex, gameState.ncols,
                gameState.nrows, gameState.opacityOff
                );
                console.log(color, '2')
                gsap.to(light, { 
                  backgroundColor: color,
                  ease: "power4",
                  boxShadow: `0 0 0px ${color}`
                });
              }
          })
        })
        if (gridSum(gameState.gridState) == gameState.nrows * gameState.ncols) {
          // window.alert("Congrtulations!")
        }
      }

      function updateState(light) {
        // always flip the light that was switched
        let row = parseInt(light.dataset.row)
        let col = parseInt(light.dataset.col)
        gameState.gridState[row][col] = !gameState.gridState[row][col]

        // record whether the light is being turned on (0 -> 1) or off (1 --> 0)
        let delta = gameState.gridState[row][col]

        for (const [key, concept] of Object.entries(gameDynamics)) {
          if (concept.triggers(row, col)) {
            concept.apply({col:col, row:row}, delta)
          }
        }
        // always render state
        renderState()
      }

      function getcolor(row, col, nrows, ncols, opacity) {
        if (gameDynamics[0].targets(row, col)) {
          return `rgba(195, 4, 209, ${opacity})`
        } else if (gameDynamics[0].triggers(row, col)) {
          return `rgba(102, 0, 255, ${opacity})`
        } else {
          return `rgba(153, 204, 0, ${opacity})`
        }

        }

      function createLight(col, row, ncols, nrows) {
        // create the light
        let light = document.createElement('span');
        light.classList.add('light');
        light.dataset.col = col;
        light.dataset.row = row;
        light.id = `light_${row}_${col}`
        light.onclick = function() {
          updateState(this)
        }
        light.setAttribute("id", `light_${row}_${col}`);
        let color = getcolor(row, col, nrows, ncols, gameState.opacityOff) 
        light.style.backgroundColor = color;
        // light.style.borderColor = color;
        // light.onmouseenter = function(){
        //   light.style.borderColor = "black";
        // };
        // light.onmouseleave = function(){
        //   light.style.borderColor = "#505250";
        // };
        return light
      }

      function createGrid(nrows, ncols) {
        // create the grid
        let grid = document.getElementById("gameGrid")
        document.documentElement.style.setProperty("--ncolumns", ncols);
        document.documentElement.style.setProperty("--nrows", nrows);
        
        for (let row = 0; row < nrows; row++) {
          let rowArray = new Array(ncols) 
          for (let col = 0; col < ncols; col++) {
            
            // create a holder for a light
            let slot = document.createElement('div');
            slot.classList.add('grid-item');
            
            // create a light
            let light = createLight(col, row, ncols, nrows)
            
            // add the light to the game
            rowArray[col] = false

            // add both to DOM
            slot.appendChild(light)
            grid.appendChild(slot)
          }
          gameState.gridState.push(rowArray)
        }

      }

      // partition function builders
      function rowBuilder(selectedRow){
        return function (rowIndex, colIndex) {return rowIndex == selectedRow}
      }
      function colBuilder(selectedCol){
        return function (rowIndex, colIndex) {return colIndex == selectedCol}
      }

      // partition functions
      function topRow (rowIndex, colIndex) {return rowBuilder(0)}
      function bottomRow (rowIndex, colIndex) {return rowBuilder(gameState.ncols - 1)}
      function leftCol (rowIndex, colIndex) {return colBuilder(0)}
      function rightCol (rowIndex, colIndex) {return colBuilder(gameState.nrows - 1)}
      function mainDiagonal (rowIndex, colIndex) {return function (rowIndex, colIndex) {return rowIndex==colIndex}}
      function upperLeftL (rowIndex, colIndex) {return function (rowIndex, colIndex) {return (rowIndex==0) || (colIndex==0)}}
      let partititonFunctions = [
        topRow,
        bottomRow,
        leftCol,
        rightCol,
        mainDiagonal,
        upperLeftL
      ];

      function createBooleanArray(f, nrows, ncols) {
        let myFunc = f(nrows, ncols)
        let a = [];
        for (let rowIndex = 0; rowIndex < nrows; rowIndex++) {
          let row = [];
          for (let colIndex = 0; colIndex < ncols; colIndex++) {
            row[colIndex] = myFunc(rowIndex, colIndex)
          }
          a.push(row)
        }
        return a
      }

      function turnOn(rowIndex, colIndex){
        return true
      }
      function turnOff(rowIndex, colIndex){
        return false
      }
      function flip(rowIndex, colIndex){
        return !(gameState.gridState[rowIndex][colIndex])
      }
      let actionFunctions = [
        // turnOn,
        turnOff,
        // flip,
      ];
      
      function sampleDynamic(nrows, ncols) {
        let dynamics = {};
        // create trigger array
        let grids = jsPsych.randomization.sampleWithoutReplacement(partititonFunctions, 2);
        dynamics['triggerGrid'] = createBooleanArray(
          grids[0],
          nrows,
          ncols
        )
        // create target array
        dynamics['targetGrid'] = createBooleanArray(
          grids[1],
          nrows,
          ncols
        )
        dynamics['triggers'] = function (row, col) {
          return (dynamics.triggerGrid[row][col] && !(dynamics.targetGrid[row][col]))
        }
        dynamics['targets'] = function (row, col) {
          return dynamics.targetGrid[row][col]
        }
        // choose an action to apply
        dynamics['actionClass'] = actionFunctions[
            Math.floor(Math.random()*actionFunctions.length)
        ]
        console.log(dynamics.actionClass)
        dynamics['action'] = function (rowIndex, colIndex, delta) { 
          let newState = undefined;

          // If we're turning the light on, we want to apply the action
          // if we're turning the light off, do we we want to un-apply the action?
          // other options: 
          // 1. when toggling a light it should just undo exactly what the change was 
          // (i.e. just the lights that changed when it was turned on change back when its turned off)
          // this can be different to the full target partition set if some were on and some were off
          // 2. Maybe turning off a light should leave the effect undone, but then turning it back on does nothing? 
          // maybe these two dimensions are orthogonal: reverse or not reverse; full set or just set that changed when efffect was applied?
          // if (delta) {
          //   // light is being turned on: apply effect
          //   newState = dynamics['actionClass'](rowIndex, colIndex)
          // } else if (gameState.actionDelta) {
          //   // light is being turned off: un-apply effect
          //   newState = !(dynamics['actionClass'](rowIndex, colIndex))
          // }
          newState = dynamics['actionClass'](rowIndex, colIndex)
          gameState.gridState[rowIndex][colIndex] = newState
        }

        dynamics['apply'] = function (ignore, delta) {
          for (let row = 0; row < gameState.nrows; row++) {
            for (let col = 0; col < gameState.ncols; col++) {
              if ((col==ignore.col) && (row==ignore.row)) {continue}
                if (this.targets(row, col)) {
                  this.action(row, col, delta)
                }
              }
            }
          }
          return dynamics
      }

      let gameDynamics = {
        0: sampleDynamic(gameState.nrows, gameState.ncols),
        // 1: sampleDynamic(gameState.nrows, gameState.ncols)
      }

      addCSS(`
        :root {
                --nrows: 2;
                --ncolumns: 2;
              }
              
              
              .grid-container {
                display: grid;
                column-gap: 1px;
                row-gap: 1px;
                border-color: #505250;
                border-style: solid;
                border-width: 0px;
                border-radius: 10%;
                padding: 1px;
                grid-template-columns: repeat(var(--ncolumns), 1fr);
                grid-template-rows: repeat(var(--nrows), 1fr);
                
              }

              .light {
                height: 25px;
                width: 25px;
                padding: 1px;
                background-color: #505250;
                border-radius: 25%;
                display: inline-block;
                border-style: solid;
                border-color: white;
                border-width: 0px;           }

              .centered {
                margin: 0;
                position: absolute;
                top: 50%;
                left: 50%;
                -ms-transform: translate(-50%, -50%);
                transform: translate(-50%, -50%);
              }

      `)

      // display the target word
      // and two buttons
      var html_content = `
        <h3>Turn on all the lights!</h3>
        <div class="grid-container" id="gameGrid"></div>
        <button>Finish</button>
      `
      display_element.innerHTML = html_content;
      
      createGrid(gameState.ncols,gameState.nrows)

    }
  }
  jsPsychLightGridPlugin.info = info;

  return jsPsychLightGridPlugin;
})(jsPsychModule);