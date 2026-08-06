/* ============================================================
   Candidate profiles

   Kept separate from app.js so biography text can be edited
   without touching application logic.
   ============================================================ */

const PROFILES = {
  'amresh-kumar-singh': {
    role: 'Member of Parliament, Sarlahi-4',
    education: 'PhD in International Relations, Jawaharlal Nehru University, New Delhi',
    background: 'Former academic and lecturer at Amrit Science Campus. Longtime Madhesi rights activist and three-term parliamentarian, known for an outspoken and independent stance in the House of Representatives and for commentary on federalism and inclusion.',
  },

  'balen': {
    role: 'Senior Leader, Rastriya Swatantra Party',
    education: 'BE Civil Engineering, Purbanchal University · MTech Structural Engineering, Visvesvaraya Technological University',
    background: 'Structural engineer and hip-hop artist, a pioneer of Nephop and founder of the Raw Barz rap battle platform. As Mayor of Kathmandu, known for municipal administration reform, heritage restoration, clearing illegal encroachment, public school upgrades and waste management.',
  },

  'gagan-thapa': {
    role: 'President, Nepali Congress · MP, Sarlahi-4',
    education: 'MA Sociology, Tribhuvan University',
    background: 'Youth leader who rose to national prominence as a Free Student Union activist during Jana Andolan II in 2006. As Minister for Health and Population he led healthcare infrastructure reform and enacted the National Health Insurance Act.',
  },

  'goma-tamang': {
    role: 'Central Leader, Rastriya Swatantra Party · Sunsari',
    education: 'School Leaving Certificate',
    background: 'Social activist, entrepreneur and former migrant worker who spent fourteen years as a caregiver in Israel. Active in rescuing victims of human trafficking and stranded overseas workers, alongside advocacy for women\u2019s empowerment, migrant rights and LGBTQIA+ representation.',
  },

  'harka-sampang': {
    role: 'President and Founder, Shram Sanskriti Party',
    education: 'BA English and Sociology, Mahendra Multiple Campus, Dharan',
    background: 'Grassroots civic campaigner who pioneered community voluntary labour (shramdaan). Mobilised thousands of local citizens to lay miles of water pipeline addressing Dharan\u2019s drinking water shortage, and led regional afforestation and anti-addiction drives.',
  },

  'kp-oli': {
    role: 'Chairman, CPN (UML) · former Prime Minister',
    education: 'School Leaving Certificate; self-taught in political philosophy and history across fourteen years of imprisonment',
    background: 'Veteran of the 1970s Jhapa Movement who served fourteen years in prison under the Panchayat regime. A key architect of the modern CPN (UML), recognised for leadership during the drafting of the 2015 Constitution, a nationalist foreign policy stance during the 2015 border blockade, and major infrastructure initiatives.',
  },

  'leelamani-gautam': {
    role: 'Central Committee Member and Rukum East District President, CPN (UML)',
    education: 'MA Public Administration and Rural Development, Tribhuvan University · LLB, Nepal Law Campus',
    background: 'Youth politician who became one of the youngest district committee presidents in CPN (UML). Rose through student politics via ANNFSU, serving as a student union leader during the 2006 democracy movement.',
  },

  'mina-kharel': {
    role: 'Mahasamiti Member, Nepali Congress · President, Nepal Women\u2019s Association (Chitwan)',
    education: 'MA Sociology',
    background: 'Social activist and women\u2019s rights campaigner with over thirty-five years of service in Chitwan. Founder of Adarsha Griha, a rehabilitation shelter for victimised women and children, and Hamro Ghar, a street children rehabilitation centre, with a background in Nepal Student Union leadership and local government.',
  },

  'prachanda': {
    role: 'Chairman, CPN (Maoist Centre) · former Prime Minister',
    education: 'BSc Agriculture, Institute of Agriculture and Animal Science, Rampur',
    background: 'Led the CPN (Maoist) through the decade-long civil war from 1996 to 2006. Principal architect of the 2006 Comprehensive Peace Accord, leading the political transition that abolished the monarchy and established Nepal as a federal democratic republic.',
  },

  'rabi-lamichhane': {
    role: 'Founder and President, Rastriya Swatantra Party · MP, Chitwan-2 · former Deputy Prime Minister and Home Minister',
    education: 'Higher secondary and college education in Nepal, in journalism and humanities',
    background: 'Former investigative television journalist, best known for hosting Sidha Kura Janata Sanga and for a Guinness World Record for the longest continuous television talk show hosting. Founded the RSP in 2022 around public accountability, anti-corruption and civil service delivery reform.',
  },
};
