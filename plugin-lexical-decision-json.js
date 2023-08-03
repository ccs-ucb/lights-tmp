var jsPsychLightGrid = (function (jspsych) {
  "use strict";

  const info = {
    name: "LIGHT-GRID",
    parameters: {
      game: {
        type: jspsych.ParameterType.JSON,
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

  const shuffleArray = array => {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
  }

  class jsPsychLightGridPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {

      let game = {"state":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"groups":[[2,1,0,1,0,1],[4,4,0,1,3,2],[2,4,4,2,3,2],[2,2,2,1,2,4],[3,0,4,4,4,2],[0,4,3,1,1,0]],"dag":[[0,1,1,1,1],[0,0,0,1,0],[0,0,0,1,1],[0,0,0,0,1],[0,0,0,0,0]]}
      // let game = trial.game
      console.log(game)
      
      let colors = [
        {"r":253,"g":181, "b":21},
        {"r":238,"g":31, "b":96},
        {"r":0,"g":176, "b":218},
        {"r":133,"g":148, "b":56},
        {"r":237,"g":78, "b":51},
        {"r":221,"g":213, "b":199},
      ]

      // setup the game
      let gameState = {
        gridState: game.state ,//binary 2d array
        opacityOn: .95,
        opacityOff: .25,
        ncols: game.state[0].length,
        nrows: game.state.length,
        groups: game.groups,
        dynamics: game.dynamics,
        DAG: game.dag,
        colors: colors,
        actionDelta: false // turning off a light reverses it's *action*
      }
      console.log(gameState)

      let trialData = {}

      const turnLightOn = (light, rgb) => {
        let color = `rgba(${rgb.r}, ${rgb.g},${rgb.b}, ${gameState.opacityOn})`
        gsap.to(light, { 
          backgroundColor: color,
          ease: "power4",
          boxShadow: `0 0 10px ${color}`,
          borderColor: color,
        });
      }

      const turnLightOff = (light, rgb) => {
        let bgColor = `rgba(${rgb.r}, ${rgb.g},${rgb.b}, ${gameState.opacityOff})`
        let borderColor = `rgba(${rgb.r}, ${rgb.g},${rgb.b}, ${gameState.opacityOn})`
        gsap.to(light, { 
          backgroundColor: bgColor,
          ease: "power2.out",
          boxShadow: `0 0 0px`,
          borderColor: borderColor,
        });
      }

      // const applyDynamic = (dynamic, ignore) => {
      //   for (let row = 0; row < gameState.nrows; row++) {
      //     for (let col = 0; col < gameState.ncols; col++) {
      //     if ((col==ignore.col) && (row==ignore.row)) {continue}
      //       if (dynamic.target[row][col]) {
      //         gameState.gridState[row][col] = 0
      //       }
      //     }
      //   }
      // }

      const applyDynamic = (targetGroups) => {
        for (let row = 0; row < gameState.nrows; row++) {
          for (let col = 0; col < gameState.ncols; col++) {
            if ((targetGroups.includes(gameState.groups[row][col]))) {
              gameState.gridState[row][col] = 0
            }
          }
        }
      }

      const finishGame = () => {
        gsap.to('.light', { 
          ease: "elastic.out",
          rotation: 360,
          duration: 2,
          // yoyo: true
        });
        let rgb = colors[0]
        let color = `rgba(${rgb.r}, ${rgb.g},${rgb.b}, ${gameState.opacityOn})`
        gsap.to('.grid-container', { 
          ease: "elastic.out",
          borderColor: color,
          boxShadow: `0 0 10px ${color}`,
          borderWidth: '3px',
          duration: 2,
          // yoyo: true
        });
      }      

      function renderState() {
        gameState.gridState.forEach(function (row, rowIndex) {
          row.forEach(function (col, colIndex) {
            let key = `light_${rowIndex}_${colIndex}`
            let light = document.getElementById(key)
            let group = gameState.groups[rowIndex][colIndex]
            let rgb = gameState.colors[group]
            col ? turnLightOn(light,rgb) : turnLightOff(light, rgb) 
          })
        }) 
        if (gridSum(gameState.gridState) == (gameState.nrows * gameState.ncols)) {
          finishGame()
        }
      }

      const turnOffGroup = (group) => {
        targets = document.querySelectorAll(`[data-group=${group}]`)
        let rgb = gameState.colors[group]
        targets.forEach(function(light){
          turnLightOff(light, rgb)
        })
      }

      function updateState(light) {
        // always flip the light that was switched
        let row = parseInt(light.dataset.row)
        let col = parseInt(light.dataset.col)
        gameState.gridState[row][col] = !gameState.gridState[row][col]

        // record whether the light is being turned on (0 -> 1) or off (1 --> 0)
        let delta = gameState.gridState[row][col]
        let group = gameState.groups[row][col]
        
        let targetGroups = gameState.DAG[group].flatMap((group, i) => group ? i : []);
        console.log(group, targetGroups, 'gtg')
        applyDynamic(targetGroups)
        // for (const [key, concept] of Object.entries(gameState.dynamics)) {
        //   console.log(concept)
        //   if (concept.source[row][col]) {
        //     let ignore = {col:col, row:row}
        //     applyDynamic(concept, ignore)
        //   }
        // }

        // always render state
        renderState()
      }

      function createLight(col, row) {
        // create the light
        let light = document.createElement('span');
        let group = gameState.groups[row][col]
        light.classList.add('light');
        light.dataset.col = col;
        light.dataset.row = row;
        light.dataset.group = group;
        light.id = `light_${row}_${col}`
        light.onclick = function() {
          updateState(this)
        }
        light.setAttribute("id", `light_${row}_${col}`);
        let rgb = gameState.colors[group]
        turnLightOff(light, rgb)
        return light
      }

      function createGrid(nrows, ncols) {
        // create the grid
        let grid = document.getElementById("gameGrid")
        document.documentElement.style.setProperty("--ncolumns", ncols);
        document.documentElement.style.setProperty("--nrows", nrows);
        
        for (let row = 0; row < nrows; row++) {
          
          for (let col = 0; col < ncols; col++) {
            
            // create a holder for a light
            let slot = document.createElement('div');
            slot.classList.add('grid-item');
            
            // create a light
            let light = createLight(col, row)
            
            // add both to DOM
            slot.appendChild(light)
            grid.appendChild(slot)
          }
        }

      }

      addCSS(`
        :root {
                --nrows: 2;
                --ncolumns: 2;
              }
              
              
              .grid-container {
                display: grid;
                column-gap: 10px;
                row-gap: 10px;
                border-color: #C0C0C0;
                border-style: solid;
                background-color: #25123b;
                border-width: 0px;
                border-radius: 5%;
                padding: 50px;
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
                border-color: grey;
                border-width: 2px;           }

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